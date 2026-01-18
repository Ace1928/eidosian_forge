import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def quantize_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_global[grid](x, absmax_inv, output, n_elements)
    return (output, absmax)