import numpy as np
from numpy.testing import (
def test_mem_masked_where(self):
    from numpy.ma import masked_where, MaskType
    a = np.zeros((1, 1))
    b = np.zeros(a.shape, MaskType)
    c = masked_where(b, a)
    a - c