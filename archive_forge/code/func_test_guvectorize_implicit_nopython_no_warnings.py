import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_guvectorize_implicit_nopython_no_warnings(self):
    with _catch_numba_deprecation_warnings() as w:

        @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)')
        def bar(a, b):
            a += 1
    self.assertFalse(w)