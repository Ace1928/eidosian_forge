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