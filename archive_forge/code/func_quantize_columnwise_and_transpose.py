import math
import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def quantize_columnwise_and_transpose(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty(N, M, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[1], device=x.device, dtype=torch.float16)
    P2 = int(2 ** math.ceil(math.log2(M)))
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_columnwise_and_transpose[grid](x, output, output_maxs, n_elements, M, N, BLOCK_SIZE=M, P2=P2)
    return (output, output_maxs)