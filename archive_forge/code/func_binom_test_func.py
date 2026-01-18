import warnings
import sys
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize
from scipy import stats
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
@staticmethod
def binom_test_func(x, n=None, p=0.5, alternative='two-sided'):
    x = np.atleast_1d(x).astype(np.int_)
    if len(x) == 2:
        n = x[1] + x[0]
        x = x[0]
    elif len(x) == 1:
        x = x[0]
        if n is None or n < x:
            raise ValueError('n must be >= x')
        n = np.int_(n)
    else:
        raise ValueError('Incorrect length for x.')
    result = stats.binomtest(x, n, p=p, alternative=alternative)
    return result.pvalue