import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
def test_fancy_indexing_boundscheck(self):
    self.assertIsNone(config.BOUNDSCHECK)
    a = np.arange(3)
    b = np.arange(4)
    with self.assertRaises(IndexError):
        fancy_array_access(a)
    fancy_array_access(b)
    at = typeof(a)
    rt = at.dtype[:]
    boundscheck = njit(rt(at), boundscheck=True)(fancy_array_access)
    with self.assertRaises(IndexError):
        boundscheck(a)