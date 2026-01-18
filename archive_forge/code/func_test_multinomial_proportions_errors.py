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
def test_multinomial_proportions_errors():
    for alpha in [-0.1, 0, 1, 1.1]:
        assert_raises(ValueError, multinomial_proportions_confint, [5] * 50, alpha=alpha)
    assert_raises(ValueError, multinomial_proportions_confint, np.arange(50) - 1)
    for method in ['unknown_method', 'sisok_method', 'unknown-glaz']:
        assert_raises(NotImplementedError, multinomial_proportions_confint, [5] * 50, method=method)