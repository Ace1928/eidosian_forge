import random
import time
import unittest
def test_ctor_nocache(self):
    decorator = self._makeOne(10, None)
    self.assertEqual(decorator.cache.size, 10)