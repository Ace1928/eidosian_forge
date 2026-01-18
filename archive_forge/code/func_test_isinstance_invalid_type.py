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
def test_isinstance_invalid_type(self):
    pyfunc = isinstance_usecase_invalid_type
    cfunc = jit(nopython=True)(pyfunc)
    self.assertTrue(cfunc(3.4))
    msg = 'Cannot infer numba type of python type'
    with self.assertRaises(errors.TypingError) as raises:
        cfunc(100)
    self.assertIn(msg, str(raises.exception))