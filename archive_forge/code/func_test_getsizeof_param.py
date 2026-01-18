import unittest
def test_getsizeof_param(self):
    self._test_getsizeof(self.Cache(maxsize=3, getsizeof=lambda x: x))