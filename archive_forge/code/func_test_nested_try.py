import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_nested_try(self):
    inner = self._multi_inner()

    @njit
    def udt(x, y, z):
        try:
            try:
                print('A')
                inner(x)
                print('B')
            except:
                print('C')
                inner(y)
                print('D')
        except:
            print('E')
            inner(z)
            print('F')
    with self.assertRaises(MyError) as raises:
        with captured_stdout() as stdout:
            udt(1, 2, 3)
    self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_two', 'E', 'call_three'])
    self.assertEqual(str(raises.exception), 'three')
    with captured_stdout() as stdout:
        udt(1, 0, 3)
    self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_other', 'D'])
    with captured_stdout() as stdout:
        udt(1, 2, 0)
    self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_two', 'E', 'call_other', 'F'])