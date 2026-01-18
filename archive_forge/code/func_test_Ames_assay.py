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
def test_Ames_assay(self):
    np.random.seed(42)
    data = [[101, 117, 111], [91, 90, 107], [103, 133, 121], [136, 140, 144], [190, 161, 201], [146, 120, 116]]
    data = np.array(data).T
    predicted_ranks = np.arange(1, 7)
    res = stats.page_trend_test(data, ranked=False, predicted_ranks=predicted_ranks, method='asymptotic')
    assert_equal(res.statistic, 257)
    assert_almost_equal(res.pvalue, 0.0035, decimal=4)
    res = stats.page_trend_test(data, ranked=False, predicted_ranks=predicted_ranks, method='exact')
    assert_equal(res.statistic, 257)
    assert_almost_equal(res.pvalue, 0.0023, decimal=4)