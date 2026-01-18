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
def test_binom_tost():
    ci = smprop.proportion_confint(10, 20, method='beta', alpha=0.1)
    bt = smprop.binom_tost(10, 20, *ci)
    assert_almost_equal(bt, [0.05] * 3, decimal=12)
    ci = smprop.proportion_confint(5, 20, method='beta', alpha=0.1)
    bt = smprop.binom_tost(5, 20, *ci)
    assert_almost_equal(bt, [0.05] * 3, decimal=12)
    ci = smprop.proportion_confint(np.arange(1, 20), 20, method='beta', alpha=0.05)
    bt = smprop.binom_tost(np.arange(1, 20), 20, ci[0], ci[1])
    bt = np.asarray(bt)
    assert_almost_equal(bt, 0.025 * np.ones(bt.shape), decimal=12)