import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_loop_in_try(self):
    inner = self._multi_inner()

    @njit
    def udt(x, n):
        try:
            print('A')
            for i in range(n):
                print(i)
                if i == x:
                    inner(i)
        except:
            print('B')
        return i
    with captured_stdout() as stdout:
        res = udt(3, 5)
    self.assertEqual(stdout.getvalue().split(), ['A', '0', '1', '2', '3', 'call_three', 'B'])
    self.assertEqual(res, 3)
    with captured_stdout() as stdout:
        res = udt(1, 3)
    self.assertEqual(stdout.getvalue().split(), ['A', '0', '1', 'call_one', 'B'])
    self.assertEqual(res, 1)
    with captured_stdout() as stdout:
        res = udt(0, 3)
    self.assertEqual(stdout.getvalue().split(), ['A', '0', 'call_other', '1', '2'])
    self.assertEqual(res, 2)