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
def test_ci_symmetry_array(count, method):
    n = 100
    a = proportion_confint([count, count], n, method=method)
    b = proportion_confint([n - count, n - count], n, method=method)
    assert_allclose(np.array(a), 1.0 - np.array(b[::-1]))