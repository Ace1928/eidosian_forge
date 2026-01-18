import numpy as np
from numba import jit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_overlap_with_k(self, pyfunc, min_ndim):
    self.check_overlap(pyfunc, min_ndim=min_ndim, have_k_argument=True)