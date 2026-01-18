from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('len_tol', [-1, 2])
def test_len_tol_validation(self, len_tol):
    error_msg = 'len_tol must be between 0 and 1.'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, self.bounds_stylinski_tang, len_tol=len_tol)