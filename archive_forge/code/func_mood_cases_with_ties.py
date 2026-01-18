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
def mood_cases_with_ties():
    expected_results = [(-1.76658511464992, 0.0386488678399305), (-0.694031428192304, 0.243831249864725), (-1.15093525352151, 0.124879436583615)]
    seeds = [23453254, 1298352315, 987234597]
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        xy = rng.random(100)
        tie_ind = rng.integers(low=0, high=99, size=5)
        num_ties_per_ind = rng.integers(low=1, high=5, size=5)
        for i, n in zip(tie_ind, num_ties_per_ind):
            for j in range(i + 1, i + n):
                xy[j] = xy[i]
        rng.shuffle(xy)
        x, y = np.split(xy, 2)
        yield (x, y, 'less', *expected_results[si])