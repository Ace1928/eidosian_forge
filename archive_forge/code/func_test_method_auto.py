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
def test_method_auto(self):
    x = np.arange(20)
    y = [0.5, 4.7, 13.1]
    r1 = cramervonmises_2samp(x, y, method='exact')
    r2 = cramervonmises_2samp(x, y, method='auto')
    assert_equal(r1.pvalue, r2.pvalue)
    x = np.arange(21)
    r1 = cramervonmises_2samp(x, y, method='asymptotic')
    r2 = cramervonmises_2samp(x, y, method='auto')
    assert_equal(r1.pvalue, r2.pvalue)