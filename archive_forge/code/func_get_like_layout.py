import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
def get_like_layout(tensor: torch.Tensor, memory_format: Optional[torch.memory_format]) -> torch.memory_format:
    if memory_format is torch.preserve_format or memory_format is None:
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format