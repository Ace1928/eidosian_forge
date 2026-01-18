import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_unless_cc_53
def test_mixed_fp16_binary_arithmetic(self):
    functions = (simple_fp16add, simple_fp16sub, simple_fp16mul, simple_fp16_div_scalar)
    ops = (operator.add, operator.sub, operator.mul, operator.truediv)
    types = (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
    for (fn, op), ty in itertools.product(zip(functions, ops), types):
        with self.subTest(op=op, ty=ty):
            kernel = cuda.jit(fn)
            arg1 = np.random.random(1).astype(np.float16)
            arg2 = (np.random.random(1) * 100).astype(ty)
            res_ty = np.result_type(np.float16, ty)
            got = np.zeros(1, dtype=res_ty)
            kernel[1, 1](got, arg1[0], arg2[0])
            expected = op(arg1, arg2)
            np.testing.assert_allclose(got, expected)