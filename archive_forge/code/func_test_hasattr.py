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
def test_hasattr(self):

    @njit
    def foo(x):
        return (hasattr(x, '__hash__'), hasattr(x, '__not_a_valid_attr__'))
    ty = types.int64
    for x in (1, 2.34, (5, 6, 7), typed.Dict.empty(ty, ty), typed.List.empty_list(ty), np.ones(4), 'ABC'):
        self.assertPreciseEqual(foo(x), foo.py_func(x))