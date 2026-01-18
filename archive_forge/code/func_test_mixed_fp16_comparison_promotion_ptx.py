import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_mixed_fp16_comparison_promotion_ptx(self):
    functions = (simple_fp16_gt, simple_fp16_ge, simple_fp16_lt, simple_fp16_le, simple_fp16_eq, simple_fp16_ne)
    ops = (operator.gt, operator.ge, operator.lt, operator.le, operator.eq, operator.ne)
    types_promote = (np.int16, np.int32, np.int64, np.float32, np.float64)
    opstring = {operator.gt: 'setp.gt.', operator.ge: 'setp.ge.', operator.lt: 'setp.lt.', operator.le: 'setp.le.', operator.eq: 'setp.eq.', operator.ne: 'setp.neu.'}
    opsuffix = {np.dtype('int32'): 'f64', np.dtype('int64'): 'f64', np.dtype('float32'): 'f32', np.dtype('float64'): 'f64'}
    for (fn, op), ty in itertools.product(zip(functions, ops), types_promote):
        with self.subTest(op=op, ty=ty):
            arg2_ty = np.result_type(np.float16, ty)
            args = (b1[:], f2, from_dtype(arg2_ty))
            ptx, _ = compile_ptx(fn, args, cc=(5, 3))
            ops = opstring[op] + opsuffix[arg2_ty]
            self.assertIn(ops, ptx)