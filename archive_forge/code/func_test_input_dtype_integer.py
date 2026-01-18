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
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.int32])
def test_input_dtype_integer(self, dtype):
    x_int = np.arange(8, dtype=dtype)
    x_float = np.arange(8, dtype=np.float64)
    xt_int, lmbda_int = stats.yeojohnson(x_int)
    xt_float, lmbda_float = stats.yeojohnson(x_float)
    assert_allclose(xt_int, xt_float, rtol=1e-07)
    assert_allclose(lmbda_int, lmbda_float, rtol=1e-07)