import functools
import unittest
from torch.testing._internal.inductor_utils import HAS_CUDA
@triton.jit
def mul2_inplace_kernel(ptr, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = 2 * x
    tl.store(ptr + offsets, output, mask=mask)