from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_formula_missing_cat():
    from patsy import PatsyError
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    dta = sm.datasets.grunfeld.load_pandas().data
    dta.loc[dta.index[0], 'firm'] = np.nan
    mod = ols(formula='value ~ invest + capital + firm + year', data=dta.dropna())
    res = mod.fit()
    mod2 = ols(formula='value ~ invest + capital + firm + year', data=dta)
    res2 = mod2.fit()
    assert_almost_equal(res.params.values, res2.params.values)
    assert_raises(PatsyError, ols, 'value ~ invest + capital + firm + year', data=dta, missing='raise')