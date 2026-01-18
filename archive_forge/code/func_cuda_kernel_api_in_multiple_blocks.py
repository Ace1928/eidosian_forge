import numpy as np
from numba import cuda, float64
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit('void(float64[:,:])')
def cuda_kernel_api_in_multiple_blocks(ary):
    for i in range(2):
        tx = cuda.threadIdx.x
    for j in range(3):
        ty = cuda.threadIdx.y
    sm = cuda.shared.array((2, 3), float64)
    sm[tx, ty] = 1.0
    ary[tx, ty] = sm[tx, ty]