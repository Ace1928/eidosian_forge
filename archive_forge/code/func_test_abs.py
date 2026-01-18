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
def test_abs(self):
    f = self.jit(abs_usecase)

    def check(a):
        self.assertPreciseEqual(f(a), abs(a))
    check(TD(3))
    check(TD(-4))
    check(TD(3, 'ms'))
    check(TD(-4, 'ms'))
    check(TD('NaT'))
    check(TD('NaT', 'ms'))