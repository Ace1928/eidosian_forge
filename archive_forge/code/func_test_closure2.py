import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_closure2(self):

    @njit
    def foo(x):

        def bar():
            try:
                raise ValueError('exception')
            except:
                print('CAUGHT')
                return 12
        bar()
    with captured_stdout() as stdout:
        foo(10)
    self.assertEqual(stdout.getvalue().split(), ['CAUGHT'])