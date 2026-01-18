import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
@cuda.jit(device=True)
def atomic_binary_1dim_shared2(ary, idx, op2, ary_dtype, ary_nelements, binop_func, cast_func):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(ary_nelements, ary_dtype)
    sm[tid] = ary[tid]
    cuda.syncthreads()
    bin = cast_func(idx[tid] % ary_nelements)
    binop_func(sm, bin, op2)
    cuda.syncthreads()
    ary[tid] = sm[tid]