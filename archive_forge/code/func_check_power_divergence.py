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
def check_power_divergence(self, f_obs, f_exp, ddof, axis, lambda_, expected_stat):
    f_obs = np.asarray(f_obs)
    if axis is None:
        num_obs = f_obs.size
    else:
        b = np.broadcast(f_obs, f_exp)
        num_obs = b.shape[axis]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Mean of empty slice')
        stat, p = stats.power_divergence(f_obs=f_obs, f_exp=f_exp, ddof=ddof, axis=axis, lambda_=lambda_)
        assert_allclose(stat, expected_stat)
        if lambda_ == 1 or lambda_ == 'pearson':
            stat, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof, axis=axis)
            assert_allclose(stat, expected_stat)
    ddof = np.asarray(ddof)
    expected_p = stats.distributions.chi2.sf(expected_stat, num_obs - 1 - ddof)
    assert_allclose(p, expected_p)