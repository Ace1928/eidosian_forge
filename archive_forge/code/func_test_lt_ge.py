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
def test_lt_ge(self):
    lt = self.jit(lt_usecase)
    ge = self.jit(ge_usecase)

    def check(a, b, expected):
        expected_val = expected
        not_expected_val = not expected
        if np.isnat(a) or np.isnat(a):
            expected_val = False
            not_expected_val = False
        self.assertPreciseEqual(lt(a, b), expected_val)
        self.assertPreciseEqual(ge(a, b), not_expected_val)
    check(TD(1), TD(2), True)
    check(TD(1), TD(1), False)
    check(TD(2), TD(1), False)
    check(TD(1, 's'), TD(2, 's'), True)
    check(TD(1, 's'), TD(1, 's'), False)
    check(TD(2, 's'), TD(1, 's'), False)
    check(TD(1, 'm'), TD(61, 's'), True)
    check(TD(1, 'm'), TD(60, 's'), False)
    check(TD('Nat'), TD('Nat'), False)
    check(TD('Nat', 'ms'), TD('Nat', 's'), False)
    check(TD('Nat'), TD(-2 ** 63 + 1), True)
    with self.assertRaises((TypeError, TypingError)):
        lt(TD(1, 'Y'), TD(365, 'D'))
    with self.assertRaises((TypeError, TypingError)):
        ge(TD(1, 'Y'), TD(365, 'D'))
    with self.assertRaises((TypeError, TypingError)):
        lt(TD('NaT', 'Y'), TD('NaT', 'D'))
    with self.assertRaises((TypeError, TypingError)):
        ge(TD('NaT', 'Y'), TD('NaT', 'D'))