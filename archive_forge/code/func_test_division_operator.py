import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_division_operator(self):

    @njit
    def udt(y):
        try:
            1 / y
        except Exception:
            return 57005
        else:
            return 1 / y
    self.assertEqual(udt(0), 57005)
    self.assertEqual(udt(2), 0.5)