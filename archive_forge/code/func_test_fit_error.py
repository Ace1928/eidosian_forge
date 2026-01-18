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
def test_fit_error():
    data = np.concatenate([np.zeros(29), np.ones(21)])
    message = 'Optimization converged to parameters that are...'
    with pytest.raises(FitError, match=message), pytest.warns(RuntimeWarning):
        stats.beta.fit(data)