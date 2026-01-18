import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_catch_exception(self):

    @njit
    def udt(x):
        try:
            print('A')
            if x:
                raise ZeroDivisionError('321')
            print('B')
        except Exception:
            print('C')
        print('D')
    with captured_stdout() as stdout:
        udt(True)
    self.assertEqual(stdout.getvalue().split(), ['A', 'C', 'D'])
    with captured_stdout() as stdout:
        udt(False)
    self.assertEqual(stdout.getvalue().split(), ['A', 'B', 'D'])