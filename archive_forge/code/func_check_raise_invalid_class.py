import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_invalid_class(self, cls, flags):
    pyfunc = raise_class(cls)
    cfunc = jit((types.int32,), **flags)(pyfunc)
    with self.assertRaises(TypeError) as cm:
        cfunc(1)
    self.assertEqual(str(cm.exception), 'exceptions must derive from BaseException')