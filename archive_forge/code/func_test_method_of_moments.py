import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def test_method_of_moments():
    np.random.seed(1234)
    x = [0, 0, 0, 0, 1]
    a = 1 / 5 - 2 * np.sqrt(3) / 5
    b = 1 / 5 + 2 * np.sqrt(3) / 5
    loc, scale = super(type(stats.uniform), stats.uniform).fit(x, method='MM')
    npt.assert_almost_equal(loc, a, decimal=4)
    npt.assert_almost_equal(loc + scale, b, decimal=4)