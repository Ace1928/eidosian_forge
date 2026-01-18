import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def test_array_of_object(self):
    cfunc = jit(forceobj=True)(array_of_object)
    objarr = np.array([object()] * 10)
    self.assertIs(cfunc(objarr), objarr)