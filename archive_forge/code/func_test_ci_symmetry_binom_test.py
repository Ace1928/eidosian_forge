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
@pytest.mark.parametrize('nobs', [47, 50])
@pytest.mark.parametrize('count', np.arange(48))
@pytest.mark.parametrize('array_like', [False, True])
def test_ci_symmetry_binom_test(nobs, count, array_like):
    _count = [count] * 3 if array_like else count
    nobs_m_count = [nobs - count] * 3 if array_like else nobs - count
    a = proportion_confint(_count, nobs, method='binom_test')
    b = proportion_confint(nobs_m_count, nobs, method='binom_test')
    assert_allclose(np.array(a), 1.0 - np.array(b[::-1]))