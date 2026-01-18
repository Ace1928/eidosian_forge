import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_class(self, flags):
    pyfunc = raise_class(MyError)
    cfunc = jit((types.int32,), **flags)(pyfunc)
    self.assertEqual(cfunc(0), 0)
    self.check_against_python(flags, pyfunc, cfunc, MyError, 1)
    self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
    self.check_against_python(flags, pyfunc, cfunc, np.linalg.linalg.LinAlgError, 3)