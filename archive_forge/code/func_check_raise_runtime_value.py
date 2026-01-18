import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_runtime_value(self, flags):
    pyfunc = raise_runtime_value
    cfunc = jit((types.string,), **flags)(pyfunc)
    self.check_against_python(flags, pyfunc, cfunc, ValueError, 'hello')