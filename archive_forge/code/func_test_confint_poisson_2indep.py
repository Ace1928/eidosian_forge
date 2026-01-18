import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_confint_poisson_2indep():
    count1, exposure1, count2, exposure2 = (60, 51477.5, 30, 54308.7)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='mover', compare='ratio', alpha=0.1, method_mover='jeff')
    ci1 = (1.4667, 3.0608)
    assert_allclose(ci, ci1, atol=0.05)
    ci1 = (1.466768, 3.058634)
    assert_allclose(ci, ci1, rtol=0.001)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='mover', compare='ratio', alpha=0.1, method_mover='score')
    ci1 = (1.4611, 3.0424)
    assert_allclose(ci, ci1, atol=0.05)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='waldcc', compare='ratio', alpha=0.1)
    ci1 = (1.4523, 3.0154)
    assert_allclose(ci, ci1, atol=0.0005)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='score', compare='ratio', alpha=0.05)
    ci1 = (1.365962, 3.259306)
    assert_allclose(ci, ci1, atol=5e-06)
    exposure1 /= 1000
    exposure2 /= 1000
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='mover', compare='diff', alpha=0.05, method_mover='jeff')
    ci1 = (0.2629322, 0.9786493)
    assert_allclose(ci, ci1, atol=0.005)
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method='score', compare='diff', alpha=0.05)
    ci1 = (0.265796, 0.989192)
    assert_allclose(ci, ci1, atol=5e-06)
    ci = confint_poisson_2indep(count2, exposure2, count1, exposure1, method='mover', compare='diff', alpha=0.1, method_mover='jeff')
    ci1 = (-0.9183272231752, -0.3188611692202)
    assert_allclose(ci, ci1, atol=0.005)
    ci1 = (-0.9195, -0.3193)
    assert_allclose(ci, ci1, atol=0.005)
    ci = confint_poisson_2indep(count2, exposure2, count1, exposure1, method='mover', compare='diff', alpha=0.1, method_mover='jeff')
    ci1 = (-0.9232, -0.3188)
    assert_allclose(ci, ci1, atol=0.006)