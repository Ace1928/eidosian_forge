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
@pytest.mark.parametrize('angles, ref', [([-np.pi / 2, np.pi / 2], 1.0), ([0, 2 * np.pi], 0.0)])
def test_directional_stats_2d_special_cases(self, angles, ref):
    if callable(ref):
        ref = ref(angles)
    data = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    res = 1 - stats.directional_stats(data).mean_resultant_length
    assert_allclose(res, ref)