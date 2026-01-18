import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
@functools.lru_cache(None)
def autotune_kernel(kernel: Callable):
    BLOCK_M_VALUES = [16, 32]
    BLOCK_N_VALUES = [32, 64, 128]
    STAGES_VALUES = [1, 2, 3]
    WARPS_VALUES = [1, 2, 4]
    TRITON_CONFIGS = [gen_config(block_m, block_n, stages, warps) for block_m in BLOCK_M_VALUES for block_n in BLOCK_N_VALUES for stages in STAGES_VALUES for warps in WARPS_VALUES]
    kernel = triton.autotune(configs=TRITON_CONFIGS, key=AUTOTUNER_KEY)(kernel)
    return kernel