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
def test_confint_multinomial_proportions():
    from .results.results_multinomial_proportions import res_multinomial
    for (method, description), values in res_multinomial.items():
        cis = multinomial_proportions_confint(values.proportions, 0.05, method=method)
        assert_almost_equal(values.cis, cis, decimal=values.precision, err_msg='"{}" method, {}'.format(method, description))