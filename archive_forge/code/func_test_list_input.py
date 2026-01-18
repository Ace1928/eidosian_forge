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
def test_list_input(self):
    x = [2, 3, 4, 7, 6]
    y = [0.2, 0.7, 12, 18]
    r1 = cramervonmises_2samp(x, y)
    r2 = cramervonmises_2samp(np.array(x), np.array(y))
    assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))