import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
def basic_test_all_scipy_dists(self, distname, shapes):
    slow_dists = {'ksone', 'kstwo', 'levy_stable', 'skewnorm'}
    fail_dists = {'beta', 'gausshyper', 'geninvgauss', 'ncf', 'nct', 'norminvgauss', 'genhyperbolic', 'studentized_range', 'vonmises', 'kappa4', 'invgauss', 'wald'}
    if distname in slow_dists:
        pytest.skip('Distribution is too slow')
    if distname in fail_dists:
        pytest.xfail('Fails - usually due to inaccurate CDF/PDF')
    np.random.seed(0)
    dist = getattr(stats, distname)(*shapes)
    fni = NumericalInverseHermite(dist)
    x = np.random.rand(10)
    p_tol = np.max(np.abs(dist.ppf(x) - fni.ppf(x)) / np.abs(dist.ppf(x)))
    u_tol = np.max(np.abs(dist.cdf(fni.ppf(x)) - x))
    assert p_tol < 1e-08
    assert u_tol < 1e-12