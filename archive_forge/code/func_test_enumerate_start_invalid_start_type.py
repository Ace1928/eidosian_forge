import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_enumerate_start_invalid_start_type(self):
    pyfunc = enumerate_invalid_start_usecase
    cfunc = jit((), **forceobj_flags)(pyfunc)
    with self.assertRaises(TypeError) as raises:
        cfunc()
    msg = "'float' object cannot be interpreted as an integer"
    self.assertIn(msg, str(raises.exception))