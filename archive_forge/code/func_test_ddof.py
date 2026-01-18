import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_ddof(self):
    x = np.arange(9.0)
    assert_allclose(variation(x, ddof=1), np.sqrt(60 / 8) / 4)