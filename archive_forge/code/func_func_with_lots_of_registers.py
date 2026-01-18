import numpy as np
import warnings
from numba.cuda.testing import unittest
from numba.cuda.testing import (skip_on_cudasim, skip_if_cuda_includes_missing)
from numba.cuda.testing import CUDATestCase, test_data_dir
from numba.cuda.cudadrv.driver import (CudaAPIError, Linker,
from numba.cuda.cudadrv.error import NvrtcError
from numba.cuda import require_context
from numba.tests.support import ignore_internal_warnings
from numba import cuda, void, float64, int64, int32, typeof, float32
def func_with_lots_of_registers(x, a, b, c, d, e, f):
    a1 = 1.0
    a2 = 1.0
    a3 = 1.0
    a4 = 1.0
    a5 = 1.0
    b1 = 1.0
    b2 = 1.0
    b3 = 1.0
    b4 = 1.0
    b5 = 1.0
    c1 = 1.0
    c2 = 1.0
    c3 = 1.0
    c4 = 1.0
    c5 = 1.0
    d1 = 10
    d2 = 10
    d3 = 10
    d4 = 10
    d5 = 10
    for i in range(a):
        a1 += b
        a2 += c
        a3 += d
        a4 += e
        a5 += f
        b1 *= b
        b2 *= c
        b3 *= d
        b4 *= e
        b5 *= f
        c1 /= b
        c2 /= c
        c3 /= d
        c4 /= e
        c5 /= f
        d1 <<= b
        d2 <<= c
        d3 <<= d
        d4 <<= e
        d5 <<= f
    x[cuda.grid(1)] = a1 + a2 + a3 + a4 + a5
    x[cuda.grid(1)] += b1 + b2 + b3 + b4 + b5
    x[cuda.grid(1)] += c1 + c2 + c3 + c4 + c5
    x[cuda.grid(1)] += d1 + d2 + d3 + d4 + d5