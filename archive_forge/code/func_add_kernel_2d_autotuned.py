import functools
import unittest
from torch.testing._internal.inductor_utils import HAS_CUDA
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 64}, num_stages=3, num_warps=8)], key=[])
@triton.jit
def add_kernel_2d_autotuned(in_ptr0, in_ptr1, out_ptr, x_elements, y_elements, BLOCK_SIZE_X: 'tl.constexpr', BLOCK_SIZE_Y: 'tl.constexpr'):
    xoffset = tl.program_id(0) * BLOCK_SIZE_X
    xindex = xoffset + tl.arange(0, BLOCK_SIZE_X)[:, None]
    xmask = xindex < x_elements
    yoffset = tl.program_id(1) * BLOCK_SIZE_Y
    yindex = yoffset + tl.arange(0, BLOCK_SIZE_Y)[None, :]
    ymask = yindex < y_elements
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + x_elements * y0), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (y0 + y_elements * x1), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr + (x1 + x_elements * y0), tmp2, xmask & ymask)