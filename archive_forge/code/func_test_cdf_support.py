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
def test_cdf_support(self):
    assert_equal(_cdf_cvm([1 / (12 * 533), 533 / 3], 533), [0, 1])
    assert_equal(_cdf_cvm([1 / (12 * (27 + 1)), (27 + 1) / 3], 27), [0, 1])