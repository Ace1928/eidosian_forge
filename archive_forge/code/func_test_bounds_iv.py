import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_bounds_iv(self):
    message = 'Bounds provided for the following unrecognized...'
    shape_bounds = {'n': (1, 10), 'p': (0, 1), '1': (0, 10)}
    with pytest.warns(RuntimeWarning, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'Each element of a `bounds` sequence must be a tuple...'
    shape_bounds = [(1, 10, 3), (0, 1)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'Each element of `bounds` must be a tuple specifying...'
    shape_bounds = [(1, 10, 3), (0, 1, 0.5)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    shape_bounds = [1, 0]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'A `bounds` sequence must contain at least 2 elements...'
    shape_bounds = [(1, 10)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'A `bounds` sequence may not contain more than 3 elements...'
    bounds = [(1, 10), (1, 10), (1, 10), (1, 10)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, bounds)
    message = 'There are no values for `p` on the interval...'
    shape_bounds = {'n': (1, 10), 'p': (1, 0)}
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'There are no values for `n` on the interval...'
    shape_bounds = [(10, 1), (0, 1)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'There are no integer values for `n` on the interval...'
    shape_bounds = [(1.4, 1.6), (0, 1)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)
    message = 'The intersection of user-provided bounds for `n`'
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data)
    shape_bounds = [(-np.inf, np.inf), (0, 1)]
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, self.data, shape_bounds)