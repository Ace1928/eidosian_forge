import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def test_direct_corr_matrix():
    mod = Factor(None, 2, corr=np.corrcoef(X.iloc[:, 1:-1], rowvar=0), smc=False)
    results = mod.fit(tol=1e-10)
    a = np.array([[0.965392158864, 0.225880658666255], [0.967587154301, 0.212758741910989], [0.929891035996, -0.000603217967568], [0.486822656362, -0.869649573289374]])
    assert_array_almost_equal(results.loadings, a, decimal=8)
    mod.endog_names = X.iloc[:, 1:-1].columns
    assert_array_equal(mod.endog_names, ['Basal', 'Occ', 'Max', 'id'])
    assert_raises(ValueError, setattr, mod, 'endog_names', X.iloc[:, :1].columns)