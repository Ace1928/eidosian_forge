import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_arrayscalar_const(self):
    pyfunc = use_arrayscalar_const
    cfunc = njit(())(pyfunc)
    self.assertEqual(pyfunc(), cfunc())