from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def test_iv2sls_r():
    mod = gmm.IV2SLS(endog, exog, instrument)
    res = mod.fit()
    n, k = exog.shape
    assert_allclose(res.params, params, rtol=1e-07, atol=1e-09)
    assert_allclose(res.bse, bse, rtol=0, atol=3e-07)
    assert not hasattr(mod, '_results')