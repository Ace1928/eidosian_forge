from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('f_min_rtol', [-1, 2])
def test_fmin_rtol_validation(self, f_min_rtol):
    error_msg = 'f_min_rtol must be between 0 and 1.'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, self.bounds_stylinski_tang, f_min_rtol=f_min_rtol, f_min=0.0)