import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = M * N * K / denominator
    try:
        machine_balance = 1000 * utils.get_device_tflops(dtype) / utils.get_gpu_dram_gbps()
    except Exception:
        return True
    machine_balance = machine_balance * 0.5
    return arithmetic_intensity > machine_balance