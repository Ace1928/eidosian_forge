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
def check_eq(a, b, expected):
    expected_val = expected
    not_expected_val = not expected
    if np.isnat(a) or np.isnat(b):
        expected_val = False
        not_expected_val = True
        self.assertFalse(le(a, b), (a, b))
        self.assertFalse(ge(a, b), (a, b))
        self.assertFalse(le(b, a), (a, b))
        self.assertFalse(ge(b, a), (a, b))
        self.assertFalse(lt(a, b), (a, b))
        self.assertFalse(gt(a, b), (a, b))
        self.assertFalse(lt(b, a), (a, b))
        self.assertFalse(gt(b, a), (a, b))
    with self.silence_numpy_warnings():
        self.assertPreciseEqual(eq(a, b), expected_val, (a, b, expected))
        self.assertPreciseEqual(eq(b, a), expected_val, (a, b, expected))
        self.assertPreciseEqual(ne(a, b), not_expected_val, (a, b, expected))
        self.assertPreciseEqual(ne(b, a), not_expected_val, (a, b, expected))
        if expected_val:
            self.assertTrue(le(a, b), (a, b))
            self.assertTrue(ge(a, b), (a, b))
            self.assertTrue(le(b, a), (a, b))
            self.assertTrue(ge(b, a), (a, b))
            self.assertFalse(lt(a, b), (a, b))
            self.assertFalse(gt(a, b), (a, b))
            self.assertFalse(lt(b, a), (a, b))
            self.assertFalse(gt(b, a), (a, b))
        self.assertPreciseEqual(a == b, expected_val)