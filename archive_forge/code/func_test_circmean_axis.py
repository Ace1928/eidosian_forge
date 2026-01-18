import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
def test_circmean_axis(self):
    x = np.array([[355, 5, 2, 359, 10, 350], [351, 7, 4, 352, 9, 349], [357, 9, 8, 358, 4, 356]])
    M1 = stats.circmean(x, high=360)
    M2 = stats.circmean(x.ravel(), high=360)
    assert_allclose(M1, M2, rtol=1e-14)
    M1 = stats.circmean(x, high=360, axis=1)
    M2 = [stats.circmean(x[i], high=360) for i in range(x.shape[0])]
    assert_allclose(M1, M2, rtol=1e-14)
    M1 = stats.circmean(x, high=360, axis=0)
    M2 = [stats.circmean(x[:, i], high=360) for i in range(x.shape[1])]
    assert_allclose(M1, M2, rtol=1e-14)