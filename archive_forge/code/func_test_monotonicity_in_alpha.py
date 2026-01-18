import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
@pytest.mark.parametrize('n', [20, 2000])
def test_monotonicity_in_alpha(self, n):
    rng = np.random.default_rng(42)
    x = rng.pareto(a=2, size=n)
    e_list = []
    alpha_seq = np.logspace(-15, np.log10(0.5), 100)
    for alpha in np.r_[0, alpha_seq, 1 - alpha_seq[:-1:-1], 1]:
        e_list.append(stats.expectile(x, alpha=alpha))
    assert np.all(np.diff(e_list) > 0)