import operator as op
from .. import units as pq
from .common import TestCase
def test_array_inequality(self):
    self.assertQuantityEqual([1, 2, 3, 4] * pq.J != [1, 22, 3, 44] * pq.J, [0, 1, 0, 1])
    self.assertQuantityEqual([1, 2, 3, 4] * pq.J != [1, 22, 3, 44] * pq.kg, [1, 1, 1, 1])
    self.assertQuantityEqual([1, 2, 3, 4] * pq.J != [1, 22, 3, 44], [0, 1, 0, 1])