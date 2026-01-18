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
@pytest.mark.parametrize('ksfunc', [stats.kstest, stats.ks_2samp])
@pytest.mark.parametrize('alternative, x6val, ref_location, ref_sign', [('greater', 5.9, 5.9, +1), ('less', 6.1, 6.0, -1), ('two-sided', 5.9, 5.9, +1), ('two-sided', 6.1, 6.0, -1)])
def test_location_sign(self, ksfunc, alternative, x6val, ref_location, ref_sign):
    x = np.arange(10, dtype=np.float64)
    y = x.copy()
    x[6] = x6val
    res = stats.ks_2samp(x, y, alternative=alternative)
    assert res.statistic == 0.1
    assert res.statistic_location == ref_location
    assert res.statistic_sign == ref_sign