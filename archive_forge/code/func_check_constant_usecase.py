import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def check_constant_usecase(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    for arg in self.values:
        self.assertPreciseEqual(pyfunc(arg), cfunc(arg))