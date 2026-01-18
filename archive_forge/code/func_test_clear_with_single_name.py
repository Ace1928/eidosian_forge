import random
import time
import unittest
def test_clear_with_single_name(self):
    maker = self._makeOne(maxsize=10)
    one = maker.lrucache(name='one')(_adder)
    two = maker.lrucache(name='two')(_adder)
    for i in range(100):
        _ = one(i)
        _ = two(i)
    self.assertEqual(len(maker._cache['one'].data), 10)
    self.assertEqual(len(maker._cache['two'].data), 10)
    maker.clear('one')
    self.assertEqual(len(maker._cache['one'].data), 0)
    self.assertEqual(len(maker._cache['two'].data), 10)