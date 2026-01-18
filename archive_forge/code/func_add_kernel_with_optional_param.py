import functools
import unittest
from torch.testing._internal.inductor_utils import HAS_CUDA
@triton.jit
def add_kernel_with_optional_param(in_ptr0, in_ptr1, out_ptr, n_elements, ARGS_PASSED: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    if ARGS_PASSED == 'two':
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
    else:
        output = x
    tl.store(out_ptr + offsets, output, mask=mask)