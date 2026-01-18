import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
def set_autotuner_cache(cache: Dict[Tuple[int], triton.Config], num_groups: int) -> None:
    _fwd_kernel_splitK_autotune[num_groups].cache = cache