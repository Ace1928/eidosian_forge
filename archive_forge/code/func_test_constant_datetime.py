import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_constant_datetime(self):

    def check(const):
        pyfunc = make_add_constant(const)
        f = jit(nopython=True)(pyfunc)
        x = TD(4, 'D')
        expected = pyfunc(x)
        self.assertPreciseEqual(f(x), expected)
    check(DT('2001-01-01'))
    check(DT('NaT', 'D'))