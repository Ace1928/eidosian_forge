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
@pytest.mark.parametrize('alternative, pval, rlow, rhigh, sign', [('two-sided', 0.325800137536, -0.814938968841, 0.99230697523, 1), ('less', 0.8370999312316, -1, 0.985600937290653, 1), ('greater', 0.1629000687684, -0.6785654158217636, 1, 1), ('two-sided', 0.325800137536, -0.992306975236, 0.81493896884, -1), ('less', 0.1629000687684, -1.0, 0.6785654158217636, -1), ('greater', 0.8370999312316, -0.985600937290653, 1.0, -1)])
def test_basic_example(self, alternative, pval, rlow, rhigh, sign):
    x = [1, 2, 3, 4]
    y = np.array([0, 1, 0.5, 1]) * sign
    result = stats.pearsonr(x, y, alternative=alternative)
    assert_allclose(result.statistic, 0.6741998624632421 * sign, rtol=1e-12)
    assert_allclose(result.pvalue, pval, rtol=1e-06)
    ci = result.confidence_interval()
    assert_allclose(ci, (rlow, rhigh), rtol=1e-06)