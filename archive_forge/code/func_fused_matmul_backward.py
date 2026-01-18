from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (
def fused_matmul_backward(grad_out: torch.Tensor, inputs: torch.Tensor, act_in: Optional[torch.Tensor], weight: torch.Tensor, trainable_weight: bool, trainable_bias: bool, activation_grad: int=0):
    """
    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    """
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, -2)
    inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, -2)
    assert grad_out_.shape[1] == weight.shape[0], 'Incompatible dimensions in between grad_out and weight'
    M, N = grad_out_.shape
    N, _ = weight.shape
    if activation_grad > 0:
        grad_act = torch.empty_like(grad_out_)
        if act_in is None:
            act_in = grad_out_
        grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))
        kernel_bw[grid](grad_act, grad_out_, act_in, N, grad_act.stride(0), act_in.stride(0), ACTIVATION_GRAD=activation_grad)
        grad_out_ = grad_act
    grad_in = triton.ops.matmul(grad_out_, weight)
    grad_weight = grad_out_.transpose(1, 0) @ inputs_ if trainable_weight else None
    grad_bias = torch.sum(grad_out_, dim=0) if trainable_bias else None
    return (grad_in.reshape_as(inputs), grad_weight, grad_bias)