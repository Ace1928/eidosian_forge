import os
import numpy as np
from numpy.testing import (
def test_mgrid_single_element(self):
    assert_array_equal(np.mgrid[0:0:1j], [0])
    assert_array_equal(np.mgrid[0:0], [])