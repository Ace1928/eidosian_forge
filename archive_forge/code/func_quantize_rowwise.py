import math
import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def quantize_rowwise(x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)
    P2 = int(2 ** math.ceil(math.log2(x.shape[1])))
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return (output, output_maxs)