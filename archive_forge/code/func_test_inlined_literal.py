import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
def test_inlined_literal(self):

    @njit
    def foo(a, b):
        v = 1000
        return a + literally(v) + literally(b)
    got = foo(1, 2)
    self.assertEqual(got, foo.py_func(1, 2))

    @njit
    def bar():
        a = 100
        b = 9
        return foo(a=b, b=a)
    got = bar()
    self.assertEqual(got, bar.py_func())