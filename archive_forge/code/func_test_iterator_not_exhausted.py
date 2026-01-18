import sys
import unittest
from libcloud.common.types import LazyList
def test_iterator_not_exhausted(self):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ll = LazyList(get_more=self._get_more_not_exhausted)
    number_of_iterations = 0
    for i, d in enumerate(ll):
        self.assertEqual(d, data[i])
        number_of_iterations += 1
    self.assertEqual(number_of_iterations, 10)