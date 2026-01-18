import os
import numpy as np
from numpy.testing import (
def test_histogramdd_too_many_bins(self):
    assert_raises(ValueError, np.histogramdd, np.ones((1, 10)), bins=2 ** 10)