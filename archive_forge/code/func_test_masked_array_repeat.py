import numpy as np
from numpy.testing import (
def test_masked_array_repeat(self):
    np.ma.array([1], mask=False).repeat(10)