import unittest
import numpy as np
from .. import units as pq
from .. import quantity
from .common import TestCase
def test_iec_aliases(self):
    prefixes = ['kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi']
    for prefix in prefixes:
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'bytes'))
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octet'))
        self.assertQuantityEqual(pq.B.rescale(prefix + 'byte'), pq.B.rescale(prefix + 'octets'))