import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_fp16_int8_comparison_ptx(self):
    functions = (simple_fp16_gt, simple_fp16_ge, simple_fp16_lt, simple_fp16_le, simple_fp16_eq, simple_fp16_ne)
    ops = (operator.gt, operator.ge, operator.lt, operator.le, operator.eq, operator.ne)
    opstring = {operator.gt: 'setp.gt.f16', operator.ge: 'setp.ge.f16', operator.lt: 'setp.lt.f16', operator.le: 'setp.le.f16', operator.eq: 'setp.eq.f16', operator.ne: 'setp.ne.f16'}
    for fn, op in zip(functions, ops):
        with self.subTest(op=op):
            args = (b1[:], f2, from_dtype(np.int8))
            ptx, _ = compile_ptx(fn, args, cc=(5, 3))
            self.assertIn(opstring[op], ptx)