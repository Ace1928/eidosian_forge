import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def compile_scalar_func(pyfunc, argtypes, restype):
    assert not any((isinstance(tp, types.Array) for tp in argtypes))
    assert not isinstance(restype, types.Array)
    device_func = cuda.jit(restype(*argtypes), device=True)(pyfunc)
    kernel_types = [types.Array(tp, 1, 'C') for tp in [restype] + list(argtypes)]
    if len(argtypes) == 1:

        def kernel_func(out, a):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = device_func(a[i])
    elif len(argtypes) == 2:

        def kernel_func(out, a, b):
            i = cuda.grid(1)
            if i < out.shape[0]:
                out[i] = device_func(a[i], b[i])
    else:
        assert 0
    kernel = cuda.jit(tuple(kernel_types))(kernel_func)

    def kernel_wrapper(values):
        n = len(values)
        inputs = [np.empty(n, dtype=numpy_support.as_dtype(tp)) for tp in argtypes]
        output = np.empty(n, dtype=numpy_support.as_dtype(restype))
        for i, vs in enumerate(values):
            for v, inp in zip(vs, inputs):
                inp[i] = v
        args = [output] + inputs
        kernel[int(math.ceil(n / 256)), 256](*args)
        return list(output)
    return kernel_wrapper