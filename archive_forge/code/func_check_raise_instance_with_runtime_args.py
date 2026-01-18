import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_instance_with_runtime_args(self, flags):
    for clazz in [MyError, UDEArgsToSuper, UDENoArgSuper]:
        pyfunc = raise_instance_runtime_args(clazz)
        cfunc = jit((types.int32, types.string), **flags)(pyfunc)
        self.assertEqual(cfunc(0, 'test'), 0)
        self.check_against_python(flags, pyfunc, cfunc, clazz, 1, 'hello')
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2, 'world')
        self.check_against_python(flags, pyfunc, cfunc, np.linalg.linalg.LinAlgError, 3, 'linalg')