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
def test_different_results(self):
    count1, count2 = (10000, 10000)
    nobs1, nobs2 = (10000, 10000)
    res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
    assert_allclose(res.pvalue, 1)