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
def test_int_check():
    with pytest.raises(ValueError):
        proportion_confint(10.5, 20, method='binom_test')
    with pytest.raises(ValueError):
        proportion_confint(10, 20.5, method='binom_test')
    with pytest.raises(ValueError):
        proportion_confint(np.array([10.3]), 20, method='binom_test')
    a = proportion_confint(21.0, 47, method='binom_test')
    b = proportion_confint(21, 47, method='binom_test')
    c = proportion_confint(21, 47.0, method='binom_test')
    assert_allclose(a, b)
    assert_allclose(a, c)