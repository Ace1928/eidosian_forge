import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.parametrize('kernel', ['biw', 'cos', 'epa', 'gau', 'tri', 'triw', 'uni'])
def test_all_kernels(kernel, reset_randomstate):
    data = np.random.normal(size=200)
    x_grid = np.linspace(min(data), max(data), 200)
    density = sm.nonparametric.KDEUnivariate(data)
    density.fit(kernel='gau', fft=False)
    assert isinstance(density.evaluate(x_grid), np.ndarray)