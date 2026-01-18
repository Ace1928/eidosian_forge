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
def test_callable_cdf(self):
    x, args = (np.arange(5), (1.4, 0.7))
    r1 = cramervonmises(x, distributions.expon.cdf)
    r2 = cramervonmises(x, 'expon')
    assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))
    r1 = cramervonmises(x, distributions.beta.cdf, args)
    r2 = cramervonmises(x, 'beta', args)
    assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))