import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def assert_1d_input(self, cfunc, args):
    msg = 'Input must be a 1d array.'
    self.assert_error(cfunc, args, msg)