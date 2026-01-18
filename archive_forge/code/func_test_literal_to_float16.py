import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
@skip_unless_cc_53
def test_literal_to_float16(self):
    cudafuncs = (cuda_int_literal_to_float16, cuda_float_literal_to_float16)
    hostfuncs = (reference_int_literal_to_float16, reference_float_literal_to_float16)
    for cudafunc, hostfunc in zip(cudafuncs, hostfuncs):
        with self.subTest(func=cudafunc):
            cfunc = self._create_wrapped(cudafunc, np.float16, np.float16)
            self.assertEqual(cfunc(321), hostfunc(321))