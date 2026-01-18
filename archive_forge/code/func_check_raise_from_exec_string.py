import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_from_exec_string(self, flags):
    simple_raise = "def f(a):\n  raise exc('msg', 10)"
    assert_raise = 'def f(a):\n  assert a != 1'
    for f_text, exc in [(assert_raise, AssertionError), (simple_raise, UDEArgsToSuper), (simple_raise, UDENoArgSuper)]:
        loc = {}
        exec(f_text, {'exc': exc}, loc)
        pyfunc = loc['f']
        cfunc = jit((types.int32,), **flags)(pyfunc)
        self.check_against_python(flags, pyfunc, cfunc, exc, 1)