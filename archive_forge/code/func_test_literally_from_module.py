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
def test_literally_from_module(self):

    @njit
    def foo(x):
        return numba.literally(x)
    got = foo(123)
    self.assertEqual(got, foo.py_func(123))
    self.assertIsInstance(foo.signatures[0][0], types.Literal)