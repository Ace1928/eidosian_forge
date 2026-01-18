import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('compare, meth', [('ratio', meth) for meth in method_names_poisson_2indep['confint']['ratio']] + [('diff', meth) for meth in method_names_poisson_2indep['confint']['diff']])
def test_confint(self, meth, compare):
    count1, n1, count2, n2 = (60, 514.775, 40, 543.087)
    if compare == 'ratio':
        ci_val = [1.04, 2.34]
    else:
        ci_val = [0.0057, 0.081]
    ci = confint_poisson_2indep(count1, n1, count2, n2, method=meth, compare=compare, alpha=0.05)
    assert_allclose(ci, ci_val, rtol=0.1)