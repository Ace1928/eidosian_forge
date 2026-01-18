import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.xfail(reason='geninvgauss CDF is not accurate')
def test_geninvgauss_uerror():
    dist = stats.geninvgauss(3.2, 1.5)
    rng = FastGeneratorInversion(dist)
    err = rng.evaluate_error(size=10000, random_state=67982)
    assert err[0] < 1e-10