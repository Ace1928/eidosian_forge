import triton
import triton.language as tl
@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)