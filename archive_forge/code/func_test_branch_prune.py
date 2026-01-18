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
def test_branch_prune(self):

    @njit
    def foo(x):
        if isinstance(x, str):
            return x + 'some_string'
        elif isinstance(x, complex):
            return np.imag(x)
        elif isinstance(x, tuple):
            return len(x)
        else:
            assert 0
    for x in ('string', 1 + 2j, ('a', 3, 4j)):
        expected = foo.py_func(x)
        got = foo(x)
        self.assertEqual(got, expected)