import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_inplace_conversion(self):
    for u in ('ft', 'feet', pq.ft):
        q = 10 * pq.m
        q.units = u
        self.assertQuantityEqual(q, 32.80839895 * pq.ft)