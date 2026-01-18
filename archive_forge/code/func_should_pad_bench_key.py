import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def should_pad_bench_key(mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor]=None) -> str:

    def tensor_key(t):
        return (t.shape, t.stride(), t.dtype)
    tf32_key = None if mat1.dtype != torch.float32 else torch.backends.cuda.matmul.allow_tf32
    key = (tensor_key(mat1), tensor_key(mat2), op, input if input is None else tensor_key(input), tf32_key)
    return str(key)