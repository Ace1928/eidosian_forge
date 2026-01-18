import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    M, N = input.shape
    out = torch.empty(N, M, device='cuda', dtype=torch.int8)
    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _quantize_global_transpose[grid](input, absmax_inv, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
    return (out, absmax)