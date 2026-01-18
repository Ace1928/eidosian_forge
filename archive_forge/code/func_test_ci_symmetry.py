import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
@pytest.mark.parametrize('count', np.arange(10, 90, 5))
@pytest.mark.parametrize('method', list(probci_methods.keys()) + ['binom_test'])
@pytest.mark.parametrize('array_like', [False, True])
def test_ci_symmetry(count, method, array_like):
    _count = [count] * 3 if array_like else count
    n = 100
    a = proportion_confint(count, n, method=method)
    b = proportion_confint(n - count, n, method=method)
    assert_allclose(np.array(a), 1.0 - np.array(b[::-1]))