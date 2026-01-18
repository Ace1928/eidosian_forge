import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_disable_nrt(self):

    @njit(_nrt=False)
    def raise_with_no_nrt(i):
        raise ValueError(i)
    msg = 'NRT required but not enabled'
    with self.assertRaisesRegex(errors.NumbaRuntimeError, msg):
        raise_with_no_nrt(123)