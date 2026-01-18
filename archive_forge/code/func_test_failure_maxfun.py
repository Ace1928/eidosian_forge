from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('locally_biased', [True, False])
def test_failure_maxfun(self, locally_biased):
    maxfun = 100
    result = direct(self.styblinski_tang, self.bounds_stylinski_tang, maxfun=maxfun, locally_biased=locally_biased)
    assert result.success is False
    assert result.status == 1
    assert result.nfev >= maxfun
    message = f'Number of function evaluations done is larger than maxfun={maxfun}'
    assert result.message == message