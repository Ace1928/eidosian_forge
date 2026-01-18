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
@pytest.mark.parametrize('c1, n1, c2, n2, p_expect, alt, d', ([20, 10, 20, 10, 0.999999756892963, 'two-sided', 0], [10, 10, 10, 10, 0.9999998403241203, 'two-sided', 0], [50, 15, 1, 1, 0.09920321053409643, 'two-sided', 0.05], [3, 100, 20, 300, 0.12202725450896404, 'two-sided', 0], [3, 12, 4, 20, 0.40416087318539173, 'greater', 0], [4, 20, 3, 100, 0.008053640402974236, 'greater', 0], [4, 20, 3, 10, 0.3083216325432898, 'less', 0], [1, 1, 50, 15, 0.09322998607245102, 'less', 0]))
def test_fortran_authors(self, c1, n1, c2, n2, p_expect, alt, d):
    res = stats.poisson_means_test(c1, n1, c2, n2, alternative=alt, diff=d)
    assert_allclose(res.pvalue, p_expect, atol=2e-06, rtol=1e-16)