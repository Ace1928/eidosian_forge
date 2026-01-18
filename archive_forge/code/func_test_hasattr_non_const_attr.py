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
def test_hasattr_non_const_attr(self):

    @njit
    def foo(pred):
        if pred > 3:
            attr = '__hash__'
        else:
            attr = '__str__'
        hasattr(1, attr)
    with self.assertRaises(errors.NumbaTypeError) as raises:
        foo(6)
    msg = 'hasattr() cannot determine the type of variable "attr" due to a branch.'
    self.assertIn(msg, str(raises.exception))