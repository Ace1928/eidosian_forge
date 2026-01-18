from .. import units as pq
from .common import TestCase
import numpy as np
def test_nanstd(self):
    import numpy as np
    q0 = [[1, 2], [3, 4]] * pq.m
    q1 = [[1, 2], [3, 4], [np.nan, np.nan]] * pq.m
    self.assertQuantityEqual(q0.std(), q1.nanstd())