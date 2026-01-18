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
def test_directional_stats_list_ndarray_input(self):
    data = [[0.8660254, 0.5, 0.0], [0.8660254, -0.5, 0]]
    data_array = np.asarray(data)
    res = stats.directional_stats(data)
    ref = stats.directional_stats(data_array)
    assert_allclose(res.mean_direction, ref.mean_direction)
    assert_allclose(res.mean_resultant_length, res.mean_resultant_length)