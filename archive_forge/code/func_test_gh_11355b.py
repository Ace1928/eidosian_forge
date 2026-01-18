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
@pytest.mark.parametrize(('x', 'y', 'statistic', 'pvalue'), cases_11355)
def test_gh_11355b(self, x, y, statistic, pvalue):
    res = mannwhitneyu(x, y, method='asymptotic')
    assert_allclose(res.statistic, statistic, atol=1e-12)
    assert_allclose(res.pvalue, pvalue, atol=1e-12)