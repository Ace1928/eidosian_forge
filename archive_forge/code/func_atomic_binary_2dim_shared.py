import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
@cuda.jit(device=True)
def atomic_binary_2dim_shared(ary, op2, ary_dtype, ary_shape, binop_func, y_cast_func, neg_idx):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    sm = cuda.shared.array(ary_shape, ary_dtype)
    sm[tx, ty] = ary[tx, ty]
    cuda.syncthreads()
    bin = (tx, y_cast_func(ty))
    if neg_idx:
        bin = (bin[0] - ary_shape[0], bin[1] - ary_shape[1])
    binop_func(sm, bin, op2)
    cuda.syncthreads()
    ary[tx, ty] = sm[tx, ty]