import os
import numpy as np
from numpy.testing import (
def test_asfarray_none(self):
    assert_array_equal(np.array([np.nan]), np.asfarray([None]))