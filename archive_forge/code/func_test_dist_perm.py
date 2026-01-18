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
def test_dist_perm(self):
    np.random.seed(12345678)
    x, y = self._simulations(samps=100, dims=1, sim_type='nonlinear')
    distx = cdist(x, x, metric='euclidean')
    disty = cdist(y, y, metric='euclidean')
    stat_dist, pvalue_dist, _ = stats.multiscale_graphcorr(distx, disty, compute_distance=None, random_state=1)
    assert_approx_equal(stat_dist, 0.163, significant=1)
    assert_approx_equal(pvalue_dist, 0.001, significant=1)