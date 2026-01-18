import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_axis_none(self):
    y = variation([[0, 1], [2, 3]], axis=None)
    assert_allclose(y, np.sqrt(5 / 4) / 1.5)