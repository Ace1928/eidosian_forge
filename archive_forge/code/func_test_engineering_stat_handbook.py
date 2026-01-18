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
def test_engineering_stat_handbook(self):
    """
        Example sourced from:
        https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm
        """
    group1 = [6.9, 5.4, 5.8, 4.6, 4.0]
    group2 = [8.3, 6.8, 7.8, 9.2, 6.5]
    group3 = [8.0, 10.5, 8.1, 6.9, 9.3]
    group4 = [5.8, 3.8, 6.1, 5.6, 6.2]
    res = stats.tukey_hsd(group1, group2, group3, group4)
    conf = res.confidence_interval()
    lower = np.asarray([[0, 0, 0, -2.25], [0.29, 0, -2.93, 0.13], [1.13, 0, 0, 0.97], [0, 0, 0, 0]])
    upper = np.asarray([[0, 0, 0, 1.93], [4.47, 0, 1.25, 4.31], [5.31, 0, 0, 5.15], [0, 0, 0, 0]])
    for i, j in [(1, 0), (2, 0), (0, 3), (1, 2), (2, 3)]:
        assert_allclose(conf.low[i, j], lower[i, j], atol=0.01)
        assert_allclose(conf.high[i, j], upper[i, j], atol=0.01)