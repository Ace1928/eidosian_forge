import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
def combine_state_dicts_tp(state_dicts: List[Dict[str, torch.Tensor]], config: GPT2Config):
    """Convert the list of sharded state_dict of a GPT model with tensor parallel to
    the state_dict of a standard GPT model.

    This function is meant to be the "reverse" of shard_state_dict_tp.

    Precondition:
        - state_dicts should be ordered in the same way as the shards were created.
    """
    world_size = len(state_dicts)
    keys = state_dicts[0].keys()
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0
    assert config.hidden_size % config.n_head == 0
    headdim = config.hidden_size // config.n_head

    def combine_word_embeddings(state_dicts, state_dict, key):
        dim = 0 if state_dicts[0][key].shape[0] == vocab_size // world_size else 1
        state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_dim(state_dicts, state_dict, key, dim=-1):
        if key in state_dict:
            state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_qkv_headdim(state_dicts, state_dict, key):
        n_head = config.n_head
        n_head_kv = getattr(config, 'n_head_kv', n_head)
        if key in state_dict:
            if n_head_kv == n_head:
                xs = [rearrange(s[key], '(three d) ... -> three d ...', three=3) for s in state_dicts]
                state_dict[key] = rearrange(torch.cat(xs, dim=1), 'three d ... -> (three d) ...')
            else:
                n_head_each_rank = [get_dim_for_local_rank(n_head, world_size, local_rank) for local_rank in range(world_size)]
                n_head_kv_each_rank = [get_dim_for_local_rank(n_head_kv, world_size, local_rank) for local_rank in range(world_size)]
                xs = [rearrange(s[key], '(nheadqkv headdim) ... -> nheadqkv headdim ...', nheadqkv=rank_n_head + 2 * rank_n_head_kv, headdim=headdim) for s, rank_n_head, rank_n_head_kv in zip(state_dicts, n_head_each_rank, n_head_kv_each_rank)]
                wq = torch.cat([x[:n_head_each_rank[rank]] for rank, x in enumerate(xs)], dim=0)
                wk = torch.cat([x[n_head_each_rank[rank]:n_head_each_rank[rank] + n_head_kv_each_rank[rank]] for rank, x in enumerate(xs)], dim=0)
                wv = torch.cat([x[n_head_each_rank[rank] + n_head_kv_each_rank[rank]:] for rank, x in enumerate(xs)], dim=0)
                wqkv = torch.cat([wq, wk, wv], dim=0)
                state_dict[key] = rearrange(wqkv, 'nheadqkv headdim ... -> (nheadqkv headdim) ...')

    def combine_gated_mlp(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [rearrange(s[key], '(two d) ... -> two d ...', two=2) for s in state_dicts]
            state_dict[key] = rearrange(torch.cat(xs, dim=1), 'two d ... -> (two d) ...')
    state_dict = state_dicts[0].copy()
    combine_word_embeddings(state_dicts, state_dict, 'transformer.embeddings.word_embeddings.weight')
    if 'lm_head.weight' in state_dict:
        combine_word_embeddings(state_dicts, state_dict, 'lm_head.weight')
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        combine_dim(state_dicts, state_dict, 'transformer.embeddings.position_embeddings.weight', -1)
    mlp_combine_fn = combine_gated_mlp if config.activation_function in ['glu', 'swiglu', 'geglu'] else partial(combine_dim, dim=0)
    for i in range(config.num_hidden_layers):
        combine_qkv_headdim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.Wqkv.weight')
        combine_qkv_headdim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.Wqkv.bias')
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.out_proj.weight', -1)
        mlp_combine_fn(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc1.bias', 0)
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc2.weight', -1)
    return state_dict