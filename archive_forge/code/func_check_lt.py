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
def check_lt(a, b, expected):
    expected_val = expected
    not_expected_val = not expected
    if np.isnat(a) or np.isnat(b):
        expected_val = False
        not_expected_val = False
    with self.silence_numpy_warnings():
        lt = self.jit(lt_usecase)
        self.assertPreciseEqual(lt(a, b), expected_val, (a, b, expected))
        self.assertPreciseEqual(gt(b, a), expected_val, (a, b, expected))
        self.assertPreciseEqual(ge(a, b), not_expected_val, (a, b, expected))
        self.assertPreciseEqual(le(b, a), not_expected_val, (a, b, expected))
        if expected_val:
            check_eq(a, b, False)
        self.assertPreciseEqual(a < b, expected_val)