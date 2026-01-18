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
def test_getattr_module_obj_not_implemented(self):

    @njit
    def foo():
        return getattr(np, 'cos')(1)
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    msg = 'Returning function objects is not implemented'
    self.assertIn(msg, str(raises.exception))