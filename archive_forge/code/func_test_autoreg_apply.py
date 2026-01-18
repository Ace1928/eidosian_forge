from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
def test_autoreg_apply(ols_autoreg_result):
    res, _ = ols_autoreg_result
    y = res.model.endog
    n = y.shape[0] // 2
    y = y[:n]
    x = res.model.exog
    if x is not None:
        x = x[:n]
    res_apply = res.apply(endog=y, exog=x)
    assert 'using a different' in str(res_apply.summary())
    assert isinstance(res_apply, AutoRegResultsWrapper)
    assert_allclose(res.params, res_apply.params)
    exog_oos = None
    if res.model.exog is not None:
        exog_oos = res.model.exog[-10:]
    fcasts_apply = res_apply.forecast(10, exog=exog_oos)
    assert isinstance(fcasts_apply, np.ndarray)
    assert fcasts_apply.shape == (10,)
    res_refit = res.apply(endog=y, exog=x, refit=True)
    assert not np.allclose(res.params, res_refit.params)
    assert not np.allclose(res.llf, res_refit.llf)
    assert res_apply.fittedvalues.shape == res_refit.fittedvalues.shape
    assert not np.allclose(res_apply.llf, res_refit.llf)
    if res.model.exog is None:
        fcasts_refit = res_refit.forecast(10, exog=exog_oos)
        assert isinstance(fcasts_refit, np.ndarray)
        assert fcasts_refit.shape == (10,)
        assert not np.allclose(fcasts_refit, fcasts_apply)