import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def run_shift_binop(self, pyfunc, opname):
    opfunc = getattr(operator, opname)

    def control_signed(a, b):
        tp = self.get_numpy_signed_upcast(a, b)
        return opfunc(tp(a), tp(b))

    def control_unsigned(a, b):
        tp = self.get_numpy_unsigned_upcast(a, b)
        return opfunc(tp(a), tp(b))
    samples = self.int_samples

    def check(xt, yt, control_func):
        cfunc = njit((xt, yt))(pyfunc)
        for x in samples:
            maxshift = xt.bitwidth - 1
            for y in (0, 1, 3, 5, maxshift - 1, maxshift):
                x = self.get_typed_int(xt, x)
                y = self.get_typed_int(yt, y)
                expected = control_func(x, y)
                got = cfunc(x, y)
                msg = 'mismatch for (%r, %r) with types %s' % (x, y, (xt, yt))
                self.assertPreciseEqual(got, expected, msg=msg)
    signed_pairs = [(u, v) for u, v in self.type_pairs if u.signed]
    unsigned_pairs = [(u, v) for u, v in self.type_pairs if not u.signed]
    for xt, yt in signed_pairs:
        check(xt, yt, control_signed)
    for xt, yt in unsigned_pairs:
        check(xt, yt, control_unsigned)