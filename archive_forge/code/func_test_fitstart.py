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
@pytest.mark.parametrize('distname, shapes', cases_test_fitstart())
def test_fitstart(distname, shapes):
    dist = getattr(stats, distname)
    rng = np.random.default_rng(216342614)
    data = rng.random(10)
    with np.errstate(invalid='ignore', divide='ignore'):
        guess = dist._fitstart(data)
    assert dist._argcheck(*guess[:-2])