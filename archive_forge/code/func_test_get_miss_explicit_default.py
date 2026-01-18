import random
import time
import unittest
def test_get_miss_explicit_default(self):
    cache = self._makeOne()
    default = object()
    self.assertIs(cache.get('nonesuch', default), default)