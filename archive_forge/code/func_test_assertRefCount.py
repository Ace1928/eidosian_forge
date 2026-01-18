import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_assertRefCount(self):
    x = 55.0
    y = 66.0
    l = []
    with self.assertRefCount(x, y):
        pass
    with self.assertRaises(AssertionError) as cm:
        with self.assertRefCount(x, y):
            l.append(y)
    self.assertIn('66', str(cm.exception))