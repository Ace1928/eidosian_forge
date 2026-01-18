import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def test_concentrated_predict_sarimax():
    nobs = 30
    np.random.seed(28953)
    endog = np.random.normal(size=nobs)
    out = get_sarimax_models(endog)
    assert_allclose(out.res_conc.predict(), out.res_orig.predict())
    assert_allclose(out.res_conc.forecast(5), out.res_orig.forecast(5))
    assert_allclose(out.res_conc.predict(start=0, end=45, dynamic=10), out.res_orig.predict(start=0, end=45, dynamic=10))