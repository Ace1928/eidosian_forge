import functools
import unittest
from torch.testing._internal.inductor_utils import HAS_CUDA
@triton.jit
def indirection_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: 'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if ACTIVATION == 'mul2_inplace_kernel':
        mul2_inplace_kernel(in_ptr0, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    x = tl.load(in_ptr0 + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)