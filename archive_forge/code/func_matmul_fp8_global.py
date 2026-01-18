from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F
def matmul_fp8_global(A: torch.Tensor, B: torch.Tensor, fw_code: torch.Tensor, bw_code: torch.Tensor, out: Optional[torch.Tensor]=None, bsz: int=-1, bsz2: int=-1):
    if bsz == -1 or bsz2 == -1:
        bsz, bsz2 = get_block_sizes(A, B)
    return MatMulFP8Global.apply(A, B, out, fw_code, bw_code, bsz, bsz2)