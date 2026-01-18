import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_assert_from_exec_string_objmode(self):
    self.check_raise_from_exec_string(flags=force_pyobj_flags)