import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def addmm_pattern(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)