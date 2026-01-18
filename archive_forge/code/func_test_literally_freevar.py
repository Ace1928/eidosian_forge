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
def test_literally_freevar(self):
    import numba

    @njit
    def foo(x):
        return numba.literally(x)
    self.assertEqual(foo(123), 123)
    self.assertEqual(foo.signatures[0][0], types.literal(123))