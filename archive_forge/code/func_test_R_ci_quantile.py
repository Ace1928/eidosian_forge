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
@pytest.mark.parametrize('p, alpha, lb, ub, alternative', [[0.3, 0.95, 1.22140275816017, 1.476980793882643, 'two-sided'], [0.5, 0.9, 1.506817785112854, 1.803988415397857, 'two-sided'], [0.25, 0.95, -np.inf, 1.39096812846378, 'less'], [0.8, 0.9, 2.117000016612675, np.inf, 'greater']])
def test_R_ci_quantile(self, p, alpha, lb, ub, alternative):
    x = np.exp(np.arange(0, 1.01, 0.01))
    res = stats.quantile_test(x, p=p, alternative=alternative)
    assert_allclose(res.confidence_interval(alpha), [lb, ub], rtol=1e-15)