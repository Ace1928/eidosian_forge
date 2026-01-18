import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_bad_nan_policy(self):
    with pytest.raises(ValueError, match='must be one of'):
        variation([1, 2, 3], nan_policy='foobar')