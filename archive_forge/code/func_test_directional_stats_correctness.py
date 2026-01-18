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
def test_directional_stats_correctness(self):
    decl = -np.deg2rad(np.array([343.2, 62.0, 36.9, 27.0, 359.0, 5.7, 50.4, 357.6, 44.0]))
    incl = -np.deg2rad(np.array([66.1, 68.7, 70.1, 82.1, 79.5, 73.0, 69.3, 58.8, 51.4]))
    data = np.stack((np.cos(incl) * np.cos(decl), np.cos(incl) * np.sin(decl), np.sin(incl)), axis=1)
    dirstats = stats.directional_stats(data)
    directional_mean = dirstats.mean_direction
    mean_rounded = np.round(directional_mean, 4)
    reference_mean = np.array([0.2984, -0.1346, -0.9449])
    assert_allclose(mean_rounded, reference_mean)