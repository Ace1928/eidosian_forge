import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_53
def test_fp16_intrinsics_common(self):
    kernels = (simple_hsin, simple_hcos, simple_hlog, simple_hlog2, simple_hlog10, simple_hsqrt, simple_hceil, simple_hfloor, simple_hrcp, simple_htrunc, simple_hrint, simple_hrsqrt)
    exp_kernels = (simple_hexp, simple_hexp2)
    expected_functions = (np.sin, np.cos, np.log, np.log2, np.log10, np.sqrt, np.ceil, np.floor, np.reciprocal, np.trunc, np.rint, numpy_hrsqrt)
    expected_exp_functions = (np.exp, np.exp2)
    N = 32
    np.random.seed(1)
    x = np.random.randint(1, 65505, size=N).astype(np.float16)
    r = np.zeros_like(x)
    for kernel, fn in zip(kernels, expected_functions):
        with self.subTest(fn=fn):
            kernel = cuda.jit('void(f2[:], f2[:])')(kernel)
            kernel[1, N](r, x)
            expected = fn(x, dtype=np.float16)
            np.testing.assert_allclose(r, expected)
    x2 = np.random.randint(1, 10, size=N).astype(np.float16)
    for kernel, fn in zip(exp_kernels, expected_exp_functions):
        with self.subTest(fn=fn):
            kernel = cuda.jit('void(f2[:], f2[:])')(kernel)
            kernel[1, N](r, x2)
            expected = fn(x2, dtype=np.float16)
            np.testing.assert_allclose(r, expected)