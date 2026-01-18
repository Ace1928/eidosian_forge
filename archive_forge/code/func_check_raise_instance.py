import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_instance(self, flags):
    for clazz in [MyError, UDEArgsToSuper, UDENoArgSuper]:
        pyfunc = raise_instance(clazz, 'some message')
        cfunc = jit((types.int32,), **flags)(pyfunc)
        self.assertEqual(cfunc(0), 0)
        self.check_against_python(flags, pyfunc, cfunc, clazz, 1)
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
        self.check_against_python(flags, pyfunc, cfunc, np.linalg.linalg.LinAlgError, 3)