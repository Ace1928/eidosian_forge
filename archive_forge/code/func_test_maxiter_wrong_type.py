from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('maxiter', [1.5, 'string', (1, 2)])
def test_maxiter_wrong_type(self, maxiter):
    error_msg = 'maxiter must be of type int.'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, self.bounds_stylinski_tang, maxiter=maxiter)