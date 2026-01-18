import math
import torch
from bitsandbytes.triton.triton_utils import is_triton_available
def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    P2 = int(2 ** math.ceil(math.log2(x.shape[1])))
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _dequantize_rowwise[grid](x, state_x, output, 1.0 / 127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output