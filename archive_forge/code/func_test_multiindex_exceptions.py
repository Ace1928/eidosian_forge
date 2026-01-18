import numpy as np
from numpy.testing import (
def test_multiindex_exceptions(self):
    a = np.empty(5, dtype=object)
    assert_raises(IndexError, a.item, 20)
    a = np.empty((5, 0), dtype=object)
    assert_raises(IndexError, a.item, (0, 0))
    a = np.empty(5, dtype=object)
    assert_raises(IndexError, a.itemset, 20, 0)
    a = np.empty((5, 0), dtype=object)
    assert_raises(IndexError, a.itemset, (0, 0), 0)