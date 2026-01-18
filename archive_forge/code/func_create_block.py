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
def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    sequence_parallel = getattr(config, 'sequence_parallel', True)
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    use_rms_norm = getattr(config, 'rms_norm', False)
    norm_cls = partial(nn.LayerNorm if not use_rms_norm else RMSNorm, eps=config.layer_norm_epsilon, **factory_kwargs)
    residual_in_fp32 = getattr(config, 'residual_in_fp32', False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, 'prenorm', True)
    parallel_block = getattr(config, 'parallel_block', False)
    if not parallel_block:
        block = Block(config.hidden_size, mixer_cls, mlp_cls, norm_cls=norm_cls, prenorm=prenorm, resid_dropout1=resid_dropout1, resid_dropout2=config.resid_pdrop, fused_dropout_add_ln=getattr(config, 'fused_dropout_add_ln', False), residual_in_fp32=residual_in_fp32, sequence_parallel=sequence_parallel and process_group is not None, mark_shared_params=process_group is not None)
    else:
        assert prenorm
        block = ParallelBlock(config.hidden_size, mixer_cls, mlp_cls, norm_cls=norm_cls, resid_dropout1=resid_dropout1, resid_dropout2=config.resid_pdrop, tied_norm=getattr(config, 'parallel_block_tied_norm', False), fused_dropout_add_ln=getattr(config, 'fused_dropout_add_ln', False), residual_in_fp32=residual_in_fp32, sequence_parallel=sequence_parallel and process_group is not None, mark_shared_params=process_group is not None)
    block.layer_idx = layer_idx
    return block