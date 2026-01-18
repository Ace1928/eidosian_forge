import numpy as np
from numpy.testing import (
def test_mask_not_backmangled(self):
    a = np.ma.MaskedArray([1.0, 2.0], mask=[False, False])
    assert_(a.mask.shape == (2,))
    b = np.tile(a, (2, 1))
    assert_(a.mask.shape == (2,))
    assert_(b.shape == (2, 2))
    assert_(b.mask.shape == (2, 2))