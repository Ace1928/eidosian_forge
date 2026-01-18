import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
@pytest.mark.parametrize('use_exact_diffuse', [True, False])
def test_mle_reg(use_exact_diffuse):
    endog = np.arange(100) * 1.0
    exog = endog * 2
    endog[::2] += 0.01
    endog[1::2] -= 0.01
    with warnings.catch_warnings(record=True):
        mod1 = UnobservedComponents(endog, irregular=True, exog=exog, mle_regression=False, use_exact_diffuse=use_exact_diffuse)
        res1 = mod1.fit(disp=-1)
        mod2 = UnobservedComponents(endog, irregular=True, exog=exog, mle_regression=True, use_exact_diffuse=use_exact_diffuse)
        res2 = mod2.fit(disp=-1)
    assert_allclose(res1.regression_coefficients.filtered[0, -1], 0.5, atol=1e-05)
    assert_allclose(res2.params[1], 0.5, atol=1e-05)
    if use_exact_diffuse:
        print(res1.predicted_diffuse_state_cov)
        assert_equal(res1.nobs_diffuse, 2)
        assert_equal(res2.nobs_diffuse, 0)
    else:
        assert_equal(res1.loglikelihood_burn, 1)
        assert_equal(res2.loglikelihood_burn, 0)