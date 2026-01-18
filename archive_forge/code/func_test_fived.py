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
@pytest.mark.slow
@pytest.mark.parametrize('sim_type, obs_stat, obs_pvalue', [('linear', 0.184, 1 / 1000), ('nonlinear', 0.019, 0.117)])
def test_fived(self, sim_type, obs_stat, obs_pvalue):
    np.random.seed(12345678)
    x, y = self._simulations(samps=100, dims=5, sim_type=sim_type)
    stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
    assert_approx_equal(stat, obs_stat, significant=1)
    assert_approx_equal(pvalue, obs_pvalue, significant=1)