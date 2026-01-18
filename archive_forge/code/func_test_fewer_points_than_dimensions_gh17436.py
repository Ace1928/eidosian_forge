from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_fewer_points_than_dimensions_gh17436():
    rng = np.random.default_rng(2046127537594925772)
    rvs = rng.multivariate_normal(np.zeros(3), np.eye(3), size=5)
    message = 'Number of dimensions is greater than number of samples...'
    with pytest.raises(ValueError, match=message):
        stats.gaussian_kde(rvs)