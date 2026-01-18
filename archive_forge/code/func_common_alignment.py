import itertools
from typing import List, Optional, Set, Tuple, cast
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def common_alignment(*args):
    for div in [16, 8, 4, 2]:
        if all((a % div == 0 for a in args)):
            return div
    return 1