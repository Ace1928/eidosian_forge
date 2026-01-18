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
def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mlp_fc1_bias = getattr(config, 'mlp_fc1_bias', True)
    mlp_fc2_bias = getattr(config, 'mlp_fc2_bias', True)
    fused_mlp = getattr(config, 'fused_mlp', False)
    if fused_mlp:
        assert config.activation_function in ['gelu_new', 'gelu_fast', 'gelu_approx', 'gelu_pytorch_tanh', 'relu', 'sqrelu']
    fused_dense_sqrelu_dense = getattr(config, 'fused_dense_sqrelu_dense', False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == 'sqrelu', 'fused_dense_sqrelu_dense only supports approximate activation_function sqrelu'
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if not fused_mlp and (not fused_dense_sqrelu_dense):
        assert config.activation_function in ['gelu', 'gelu_new', 'gelu_fast', 'gelu_approx', 'gelu_pytorch_tanh', 'relu', 'sqrelu', 'glu', 'swiglu', 'geglu']
        if config.activation_function in ['glu', 'swiglu', 'geglu']:
            activation = F.sigmoid if config.activation_function == 'glu' else F.silu if config.activation_function == 'swiglu' else F.gelu
            mlp_cls = GatedMlp if process_group is None else ParallelGatedMlp
            parallel_kwargs = {'process_group': process_group, 'sequence_parallel': getattr(config, 'sequence_parallel', True)} if process_group is not None else {}
            mlp_multiple_of = getattr(config, 'mlp_multiple_of', 128)
            mlp_cls = partial(mlp_cls, hidden_features=config.n_inner, activation=activation, bias1=mlp_fc1_bias, bias2=mlp_fc2_bias, multiple_of=mlp_multiple_of, **parallel_kwargs, **factory_kwargs)
        else:
            if config.activation_function == 'relu':
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == 'sqrelu':
                activation = sqrelu_fwd
            else:
                approximate = 'tanh' if config.activation_function in ['gelu_new', 'gelu_fast', 'gelu_approx', 'gelu_pytorch_tanh'] else 'none'
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = Mlp if process_group is None else ParallelMLP
            parallel_kwargs = {'process_group': process_group, 'sequence_parallel': getattr(config, 'sequence_parallel', True)} if process_group is not None else {}
            mlp_cls = partial(mlp_cls, hidden_features=config.n_inner, activation=activation, bias1=mlp_fc1_bias, bias2=mlp_fc2_bias, **parallel_kwargs, **factory_kwargs)
    else:
        mlp_checkpoint_lvl = getattr(config, 'mlp_checkpoint_lvl', 0)
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError('fused_dense is not installed')
            activation = 'gelu_approx' if config.activation_function in ['gelu_new', 'gelu_fast', 'gelu_approx', 'gelu_pytorch_tanh'] else config.activation_function
            mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
            parallel_kwargs = {'process_group': process_group, 'sequence_parallel': getattr(config, 'sequence_parallel', True)} if process_group is not None else {}
            mlp_cls = partial(mlp_cls, hidden_features=config.n_inner, activation=activation, checkpoint_lvl=mlp_checkpoint_lvl, bias1=mlp_fc1_bias, bias2=mlp_fc2_bias, **parallel_kwargs, **factory_kwargs)
        elif fused_dense_sqrelu_dense:
            if process_group is not None:
                assert fused_mlp, 'Tensor Parallel is not implemented for FusedDenseSqreluDense'
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(FusedDenseSqreluDense, hidden_features=config.n_inner, checkpoint_lvl=mlp_checkpoint_lvl, **factory_kwargs)
        else:
            raise RuntimeError('MLP type not supported')
    return mlp_cls