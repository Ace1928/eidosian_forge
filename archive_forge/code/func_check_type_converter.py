import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_type_converter(self, tp, np_type, values):
    pyfunc = converter(tp)
    cfunc = jit(nopython=True)(pyfunc)
    if issubclass(np_type, np.integer):
        np_converter = lambda x: np_type(np.int64(x))
    else:
        np_converter = np_type
    dtype = np.dtype(np_type)
    for val in values:
        if dtype.kind == 'u' and isinstance(val, float) and (val < 0.0):
            continue
        expected = np_converter(val)
        got = cfunc(val)
        self.assertPreciseEqual(got, expected, msg='for type %s with arg %s' % (np_type, val))