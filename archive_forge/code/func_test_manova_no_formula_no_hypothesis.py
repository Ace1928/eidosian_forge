import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant
@pytest.mark.smoke
def test_manova_no_formula_no_hypothesis():
    exog = add_constant(pd.get_dummies(X[['Loc']], drop_first=True, dtype=float))
    endog = X[['Basal', 'Occ', 'Max']]
    mod = MANOVA(endog, exog)
    r = mod.mv_test()
    assert isinstance(r, MultivariateTestResults)