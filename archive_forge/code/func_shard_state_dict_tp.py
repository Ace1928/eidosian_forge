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
def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.

    This function modifies state_dict in place.
    """
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0
    n_head = config.n_head
    n_head_kv = getattr(config, 'n_head_kv', n_head)
    embed_dim = config.hidden_size
    head_dim = embed_dim // n_head

    def shard_first_dim(state_dict, key):
        if key in state_dict:
            x = state_dict[key]
            dim = x.shape[0] // world_size
            state_dict[key] = x[rank * dim:(rank + 1) * dim]

    def shard_last_dim(state_dict, key, multiple_of=1):
        if key in state_dict:
            x = state_dict[key]
            dim_each_rank = [get_dim_for_local_rank(x.size(-1), world_size, local_rank, multiple_of) for local_rank in range(world_size)]
            beg, end = tuple((sum(dim_each_rank[:pos]) for pos in (rank, rank + 1)))
            state_dict[key] = x[..., beg:end]

    def shard_gatedmlp_fc1_dim(state_dict, key):
        if key in state_dict:
            x = state_dict[key]
            dim = x.shape[0] // world_size // 2
            state_dict[key] = rearrange(rearrange(x, '(two o) ... -> two o ...', two=2)[:, rank * dim:(rank + 1) * dim], 'two o ... -> (two o) ...')

    def shard_qkv_headdim(state_dict, key):
        if key in state_dict:
            n_head_each_rank = [get_dim_for_local_rank(n_head, world_size, local_rank) for local_rank in range(world_size)]
            n_head_kv_each_rank = [get_dim_for_local_rank(n_head_kv, world_size, local_rank) for local_rank in range(world_size)]
            beg_n_head = sum(n_head_each_rank[:rank])
            end_n_head = sum(n_head_each_rank[:rank + 1])
            beg_n_head_kv = sum(n_head_kv_each_rank[:rank])
            end_n_head_kv = sum(n_head_kv_each_rank[:rank + 1])
            if n_head_kv == n_head:
                x = rearrange(state_dict[key], '(three d) ... -> three d ...', three=3)
                state_dict[key] = rearrange(x[:, beg_n_head * head_dim:end_n_head * head_dim], 'three d ... -> (three d) ...')
            else:
                x = rearrange(state_dict[key], '(nheadqkv headdim) ... -> nheadqkv headdim ...', nheadqkv=n_head + 2 * n_head_kv)
                state_dict[key] = rearrange(torch.cat([x[beg_n_head:end_n_head], x[n_head + beg_n_head_kv:n_head + end_n_head_kv], x[n_head + n_head_kv + beg_n_head_kv:n_head + n_head_kv + end_n_head_kv]], dim=0), 'nheadqkv headdim ... -> (nheadqkv headdim) ...')
    shard_first_dim(state_dict, 'transformer.embeddings.word_embeddings.weight')
    if 'lm_head.weight' in state_dict:
        shard_first_dim(state_dict, 'lm_head.weight')
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        shard_last_dim(state_dict, 'transformer.embeddings.position_embeddings.weight')
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.weight')
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.bias')
        shard_last_dim(state_dict, f'transformer.layers.{i}.mixer.out_proj.weight', multiple_of=head_dim)
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mixer.out_proj.bias', None)
        if config.activation_function in ['glu', 'swiglu', 'geglu']:
            shard_gatedmlp_fc1_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
            shard_gatedmlp_fc1_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.bias')
        else:
            shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
            shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.bias')
        shard_last_dim(state_dict, f'transformer.layers.{i}.mlp.fc2.weight')
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mlp.fc2.bias', None)
    return state_dict