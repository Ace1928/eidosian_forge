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
def test_confint_multinomial_proportions_zeros():
    ci01 = np.array([0.09364718, 0.1898413, 0.0, 0.0483581, 0.13667426, 0.2328684, 0.10124019, 0.1974343, 0.10883321, 0.2050273, 0.17210833, 0.2683024, 0.09870919, 0.1949033]).reshape(-1, 2)
    ci0 = np.array([0.09620253, 0.19238867, 0.0, 0.05061652, 0.13924051, 0.23542664, 0.10379747, 0.1999836, 0.11139241, 0.20757854, 0.17468354, 0.27086968, 0.10126582, 0.19745196]).reshape(-1, 2)
    ci0_shift = np.array([0.002531642, 0.002515247])
    p = [56, 0.1, 73, 59, 62, 87, 58]
    ci_01 = smprop.multinomial_proportions_confint(p, 0.05, method='sison_glaz')
    p = [56, 0, 73, 59, 62, 87, 58]
    ci_0 = smprop.multinomial_proportions_confint(p, 0.05, method='sison_glaz')
    assert_allclose(ci_01, ci01, atol=1e-05)
    assert_allclose(ci_0, np.maximum(ci0 - ci0_shift, 0), atol=1e-05)
    assert_allclose(ci_01, ci_0, atol=0.0005)