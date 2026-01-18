import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_bad_axis(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AxisError):
        variation(x, axis=10)