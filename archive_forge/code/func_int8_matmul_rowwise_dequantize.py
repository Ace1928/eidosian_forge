import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
    divfactor = 1.0 / (127.0 * 127.0)
    has_bias = 0 if bias is None else 1
    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], 'incompatible dimensions'
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    ACC_TYPE = tl.float32
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _int8_matmul_rowwise_dequantize[grid](a, b, c, bias, state_x, state_w, M, N, K, divfactor, has_bias, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), GROUP_M=8, ACC_TYPE=ACC_TYPE)
    return c