from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_against_published_reference(self):
    x = [1, 2, 3, 4, 6, 7, 8]
    y = [5, 9, 10, 11, 12, 13, 14]
    res = stats.bws_test(x, y, alternative='two-sided')
    assert_allclose(res.statistic, 5.132, atol=0.001)
    assert_equal(res.pvalue, 10 / 3432)