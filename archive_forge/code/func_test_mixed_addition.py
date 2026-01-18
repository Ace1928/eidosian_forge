import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
def test_mixed_addition(self):
    self.assertQuantityEqual(1 * pq.ft + 1 * pq.m, 4.280839895 * pq.ft)
    self.assertQuantityEqual(1 * pq.ft + pq.m, 4.280839895 * pq.ft)
    self.assertQuantityEqual(pq.ft + 1 * pq.m, 4.280839895 * pq.ft)
    self.assertQuantityEqual(pq.ft + pq.m, 4.280839895 * pq.ft)
    self.assertQuantityEqual(op.iadd(1 * pq.ft, 1 * pq.m), 4.280839895 * pq.ft)
    self.assertRaises(ValueError, lambda: 10 * pq.J + 3 * pq.m)
    self.assertRaises(ValueError, lambda: op.iadd(10 * pq.J, 3 * pq.m))