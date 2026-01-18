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
def test_literally_varargs(self):

    @njit
    def foo(a, *args):
        return literally(args)
    with self.assertRaises(errors.LiteralTypingError):
        foo(1, 2, 3)

    @njit
    def bar(a, b):
        foo(a, b)
    with self.assertRaises(errors.TypingError) as raises:
        bar(1, 2)
    self.assertIn('Cannot request literal type', str(raises.exception))