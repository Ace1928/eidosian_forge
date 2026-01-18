import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import statsmodels.api as sm
from statsmodels.stats import knockoff_regeffects as kr
from statsmodels.stats._knockoff import (RegressionFDR,
def test_equi():
    np.random.seed(2342)
    exog = np.random.normal(size=(10, 4))
    exog1, exog2, sl = _design_knockoff_equi(exog)
    exoga = np.concatenate((exog1, exog2), axis=1)
    gmat = np.dot(exoga.T, exoga)
    cm1 = gmat[0:4, 0:4]
    cm2 = gmat[4:, 4:]
    cm3 = gmat[0:4, 4:]
    assert_allclose(cm1, cm2, rtol=0.0001, atol=0.0001)
    assert_allclose(cm1 - cm3, np.diag(sl * np.ones(4)), rtol=0.0001, atol=0.0001)