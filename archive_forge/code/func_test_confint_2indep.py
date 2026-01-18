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
def test_confint_2indep():
    count1, nobs1 = (7, 34)
    count2, nobs2 = (1, 34)
    '\n    diff:\n    Wald 0.029 0.32 0.29\n    Agresti–Caffo 0.012 0.32 0.31\n    Newcombe hybrid score 0.019 0.34 0.32\n    Miettinen–Nurminen asymptotic score 0.028 0.34 0.31\n    Santner–Snell exact unconditional -0.069 0.41 0.48\n    Chan–Zhang exact unconditional 0.019 0.36 0.34\n    Agresti–Min exact unconditional 0.024 0.35 0.33\n\n    ratio:\n    Katz log 0.91 54 4.08\n    Adjusted log 0.92 27 3.38\n    Inverse sinh 1.17 42 3.58\n    Koopman asymptotic score 1.21 43 3.57\n    Chan–Zhang 1.22 181 5.00\n    Agresti–Min 1.15 89 4.35\n\n    odds-ratio\n    Woolf logit 0.99 74 4.31\n    Gart adjusted logit 0.98 38 3.65\n    Independence-smoothed logit 0.99 60 4.11\n    Cornfield exact conditional 0.97 397 6.01\n    Cornfield mid-p 1.19 200 5.12\n    Baptista–Pike exact conditional 1.00 195 5.28\n    Baptista–Pike mid-p 1.33 99 4.31\n    Agresti–Min exact unconditional 1.19 72 4.10\n    '
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, method='newcomb', compare='diff', alpha=0.05)
    assert_allclose(ci, [0.019, 0.34], atol=0.005)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, method='wald', compare='diff', alpha=0.05)
    assert_allclose(ci, [0.029, 0.324], atol=0.005)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, method='agresti-caffo', compare='diff', alpha=0.05)
    assert_allclose(ci, [0.012, 0.322], atol=0.005)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='diff', method='score', correction=True)
    assert_allclose(ci, [0.028, 0.343], rtol=0.03)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='ratio', method='log')
    assert_allclose(ci, [0.91, 54], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='ratio', method='log-adjusted')
    assert_allclose(ci, [0.92, 27], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='ratio', method='score', correction=False)
    assert_allclose(ci, [1.21, 43], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='or', method='logit')
    assert_allclose(ci, [0.99, 74], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='or', method='logit-adjusted')
    assert_allclose(ci, [0.98, 38], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='or', method='logit-smoothed')
    assert_allclose(ci, [0.99, 60], rtol=0.01)
    ci = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='odds-ratio', method='score', correction=True)
    assert_allclose(ci, [1.246622, 56.461576], rtol=0.01)