import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_abs_tol_parse(self):
    with self.assertRaises(ValueError):
        self.eq(np.float64(1e-17), np.float64(1e-17), abs_tol='invalid')
    with self.assertRaises(ValueError):
        self.eq(np.float64(1), np.float64(2), abs_tol=int(7))