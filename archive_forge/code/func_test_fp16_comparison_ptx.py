import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_fp16_comparison_ptx(self):
    functions = (simple_fp16_gt, simple_fp16_ge, simple_fp16_lt, simple_fp16_le, simple_fp16_eq, simple_fp16_ne)
    ops = (operator.gt, operator.ge, operator.lt, operator.le, operator.eq, operator.ne)
    opstring = ('setp.gt.f16', 'setp.ge.f16', 'setp.lt.f16', 'setp.le.f16', 'setp.eq.f16', 'setp.ne.f16')
    args = (b1[:], f2, f2)
    for fn, op, s in zip(functions, ops, opstring):
        with self.subTest(op=op):
            ptx, _ = compile_ptx(fn, args, cc=(5, 3))
            self.assertIn(s, ptx)