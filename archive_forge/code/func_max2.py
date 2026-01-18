import triton
import triton.language as tl
@triton.jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)