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
def test_directional_stats_2d(self):
    rng = np.random.default_rng(314499542280078925880191983383461625100)
    testdata = 2 * np.pi * rng.random((1000,))
    testdata_vector = np.stack((np.cos(testdata), np.sin(testdata)), axis=1)
    dirstats = stats.directional_stats(testdata_vector)
    directional_mean = dirstats.mean_direction
    directional_mean_angle = np.arctan2(directional_mean[1], directional_mean[0])
    directional_mean_angle = directional_mean_angle % (2 * np.pi)
    circmean = stats.circmean(testdata)
    assert_allclose(circmean, directional_mean_angle)
    directional_var = 1 - dirstats.mean_resultant_length
    circular_var = stats.circvar(testdata)
    assert_allclose(directional_var, circular_var)