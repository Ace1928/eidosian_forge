import unittest
def check_equal_intervals(x, y):
    self.assertIsInstance(x, Interval)
    self.assertIsInstance(y, Interval)
    self.assertEqual(x.lo, y.lo)
    self.assertEqual(x.hi, y.hi)