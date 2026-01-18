from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (
@triton.autotune(configs=[triton.Config({'BLOCK_N': 64}, num_stages=4, num_warps=2), triton.Config({'BLOCK_N': 128}, num_stages=3, num_warps=2), triton.Config({'BLOCK_N': 256}, num_stages=3, num_warps=4), triton.Config({'BLOCK_N': 512}, num_stages=3, num_warps=4), triton.Config({'BLOCK_N': 1024}, num_stages=3, num_warps=4)], key=['N'])
@triton.heuristics({'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def kernel_bw(GRAD_ACT, GRAD_OUT, ACT_INPUTS, N, stride_gom, stride_aim, BLOCK_N: tl.constexpr, EVEN_N: tl.constexpr, ACTIVATION_GRAD: tl.constexpr):
    """
    Go over all the activation inputs, compute the corresponding gradient
    """
    pid_m, pid_n = (tl.program_id(axis=0), tl.program_id(axis=1))
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    act_input_ptrs = ACT_INPUTS + pid_m * stride_aim + rn
    if EVEN_N:
        act_in = tl.load(act_input_ptrs)
    else:
        act_in = tl.load(act_input_ptrs, mask=rn < N, other=0.0)
    if ACTIVATION_GRAD == 1:
        grad_act = relu_grad(act_in)
    elif ACTIVATION_GRAD == 2:
        grad_act = leaky_relu_grad(act_in)
    elif ACTIVATION_GRAD == 3:
        grad_act = gelu_grad(act_in)
    elif ACTIVATION_GRAD == 4:
        grad_act = squared_relu_grad(act_in)
    elif ACTIVATION_GRAD == 5:
        grad_act = smelu_grad(act_in)
    elif ACTIVATION_GRAD == 6:
        grad_act = star_relu_grad(act_in)
    else:
        grad_act = act_in
    grad_out_ptrs = GRAD_OUT + pid_m * stride_gom + rn
    if EVEN_N:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=rn < N)
    grad_act *= grad_out
    grad_act_ptrs = GRAD_ACT + pid_m * stride_gom + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)