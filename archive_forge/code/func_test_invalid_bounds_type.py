from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('bounds', ['bounds', 2.0, 0])
def test_invalid_bounds_type(self, bounds):
    error_msg = 'bounds must be a sequence or instance of Bounds class'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, bounds)