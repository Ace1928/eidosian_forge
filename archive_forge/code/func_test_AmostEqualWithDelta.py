import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_AmostEqualWithDelta(self):
    self.assertAlmostEqual(1.1, 1.0, delta=0.5)
    self.assertAlmostEqual(1.0, 1.1, delta=0.5)
    self.assertNotAlmostEqual(1.1, 1.0, delta=0.05)
    self.assertNotAlmostEqual(1.0, 1.1, delta=0.05)
    self.assertAlmostEqual(1.0, 1.0, delta=0.5)
    self.assertRaises(self.failureException, self.assertNotAlmostEqual, 1.0, 1.0, delta=0.5)
    self.assertRaises(self.failureException, self.assertAlmostEqual, 1.1, 1.0, delta=0.05)
    self.assertRaises(self.failureException, self.assertNotAlmostEqual, 1.1, 1.0, delta=0.5)
    self.assertRaises(TypeError, self.assertAlmostEqual, 1.1, 1.0, places=2, delta=2)
    self.assertRaises(TypeError, self.assertNotAlmostEqual, 1.1, 1.0, places=2, delta=2)
    first = datetime.datetime.now()
    second = first + datetime.timedelta(seconds=10)
    self.assertAlmostEqual(first, second, delta=datetime.timedelta(seconds=20))
    self.assertNotAlmostEqual(first, second, delta=datetime.timedelta(seconds=5))