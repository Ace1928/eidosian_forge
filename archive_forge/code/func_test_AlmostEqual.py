import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_AlmostEqual(self):
    self.assertAlmostEqual(1.00000001, 1.0)
    self.assertNotAlmostEqual(1.0000001, 1.0)
    self.assertRaises(self.failureException, self.assertAlmostEqual, 1.0000001, 1.0)
    self.assertRaises(self.failureException, self.assertNotAlmostEqual, 1.00000001, 1.0)
    self.assertAlmostEqual(1.1, 1.0, places=0)
    self.assertRaises(self.failureException, self.assertAlmostEqual, 1.1, 1.0, places=1)
    self.assertAlmostEqual(0, 0.1 + 0.1j, places=0)
    self.assertNotAlmostEqual(0, 0.1 + 0.1j, places=1)
    self.assertRaises(self.failureException, self.assertAlmostEqual, 0, 0.1 + 0.1j, places=1)
    self.assertRaises(self.failureException, self.assertNotAlmostEqual, 0, 0.1 + 0.1j, places=0)
    self.assertAlmostEqual(float('inf'), float('inf'))
    self.assertRaises(self.failureException, self.assertNotAlmostEqual, float('inf'), float('inf'))