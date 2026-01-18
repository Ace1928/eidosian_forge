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
def test_mood_2d(self):
    ny = 5
    np.random.seed(1234)
    x1 = np.random.randn(10, ny)
    x2 = np.random.randn(15, ny)
    z_vectest, pval_vectest = stats.mood(x1, x2)
    for j in range(ny):
        assert_array_almost_equal([z_vectest[j], pval_vectest[j]], stats.mood(x1[:, j], x2[:, j]))
    x1 = x1.transpose()
    x2 = x2.transpose()
    z_vectest, pval_vectest = stats.mood(x1, x2, axis=1)
    for i in range(ny):
        assert_array_almost_equal([z_vectest[i], pval_vectest[i]], stats.mood(x1[i, :], x2[i, :]))