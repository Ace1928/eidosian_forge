import math
import warnings
from numba import jit
from numba.core.errors import TypingError, NumbaWarning
from numba.tests.support import TestCase
import unittest
def test_mutual_2(self):
    from numba.tests.recursion_usecases import make_mutual2
    pfoo, pbar = make_mutual2()
    cfoo, cbar = make_mutual2(jit(nopython=True))
    for x in [-1, 0, 1, 3]:
        self.assertPreciseEqual(pfoo(x=x), cfoo(x=x))
        self.assertPreciseEqual(pbar(y=x, z=1), cbar(y=x, z=1))