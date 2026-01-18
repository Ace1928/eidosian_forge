import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def our_estimate_matmul_time(A11, B11, C11, M1, M2, M3, N1, N2, N3, K1, K2, K3, **kwargs):
    """Call into Triton's upstream cost model, with the right args

    The upstream function expects arguments to have certain names. Since we
    renamed a few of them in our implementation, we rename them back.

    At the time of writing (July 2023) the arguments that Triton expects are:
    M, N, K, A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages.

    """
    return estimate_matmul_time(M=M1 + M2 + M3, N=N1 + N2 + N3, K=K1 + K2 + K3, A=A11, B=B11, C=C11, **kwargs)