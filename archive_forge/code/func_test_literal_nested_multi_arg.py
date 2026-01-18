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
def test_literal_nested_multi_arg(self):

    @njit
    def foo(a, b, c):
        return inner(a, c)

    @njit
    def inner(x, y):
        return x + literally(y)
    kwargs = dict(a=1, b=2, c=3)
    got = foo(**kwargs)
    expect = (lambda a, b, c: a + c)(**kwargs)
    self.assertEqual(got, expect)
    [foo_sig] = foo.signatures
    self.assertEqual(foo_sig[2], types.literal(3))