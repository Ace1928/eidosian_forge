import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
def test_no_penalization(self):
    res_ols = OLS(self.res1.model.endog, self.res1.model.exog).fit()
    res_theil = self.res1.model.fit(pen_weight=0, cov_type='data-prior')
    assert_allclose(res_theil.params, res_ols.params, rtol=1e-10)
    assert_allclose(res_theil.bse, res_ols.bse, rtol=1e-10)