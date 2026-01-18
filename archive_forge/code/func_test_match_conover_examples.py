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
def test_match_conover_examples(self):
    x = [189, 233, 195, 160, 212, 176, 231, 185, 199, 213, 202, 193, 174, 166, 248]
    pvalue_expected = 0.0346
    res = stats.quantile_test(x, q=193, p=0.75, alternative='two-sided')
    assert_allclose(res.pvalue, pvalue_expected, rtol=1e-05)
    x = [59] * 8 + [61] * (112 - 8)
    pvalue_expected = stats.binom(p=0.5, n=112).pmf(k=8)
    res = stats.quantile_test(x, q=60, p=0.5, alternative='greater')
    assert_allclose(res.pvalue, pvalue_expected, atol=1e-10)