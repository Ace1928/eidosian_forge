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
def test_getattr_no_optional_type_generated(self):

    @njit
    def default_hash():
        return 12345

    @njit
    def foo():
        hash_func = getattr(np.ones(1), '__not_a_valid_attr__', default_hash)
        return hash_func()
    self.assertPreciseEqual(foo(), foo.py_func())