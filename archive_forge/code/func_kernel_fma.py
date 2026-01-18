from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (
@triton.autotune(configs=[c for block_k in [32, 64] for c in get_configs(block_k)], key=['M', 'N', 'K'])
@triton.heuristics({'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def kernel_fma(OUT, ACT_INPUTS, INPUT, WEIGHT, bias, M, N, K, stride_om, stride_im, stride_wn, BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_N: tl.constexpr, BIAS: tl.constexpr, SAVE_ACT_INPUTS: tl.constexpr, ACTIVATION: tl.constexpr, is_fp16: tl.constexpr):
    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % GROUP_M
    pid_n = pid % num_pid_in_group // GROUP_M
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    input_ptrs = INPUT + rm[:, None] * stride_im
    weight_ptrs = WEIGHT + rn[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if BIAS:
        if EVEN_N:
            bias = tl.load(bias + rn).to(tl.float32)
        else:
            bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    mask_rn = rn < N
    mask_rm = rm < M
    for i in range(0, K, BLOCK_K):
        rk = tl.arange(0, BLOCK_K) + i
        a = tl.load(input_ptrs + rk[None, :], mask=(rk[None, :] < K) & mask_rm[:, None], other=0.0)
        w = tl.load(weight_ptrs + rk[:, None], mask=(rk[:, None] < K) & mask_rn[None, :], other=0.0)
        acc += tl.dot(a, w)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + rm[:, None] * stride_om + rn[None, :]
        tl.store(act_in_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])
    if ACTIVATION == 1:
        acc = relu(acc)
    elif ACTIVATION == 2:
        acc = leaky_relu(acc)
    elif ACTIVATION == 3:
        acc = gelu(acc)
    elif ACTIVATION == 4:
        acc = squared_relu(acc)
    elif ACTIVATION == 5:
        acc = smelu(acc)
    elif ACTIVATION == 6:
        acc = star_relu(acc)
    out_ptrs = OUT + rm[:, None] * stride_om + rn[None, :]
    tl.store(out_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])