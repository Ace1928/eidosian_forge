from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_append_extend_apply_invalid():
    niledata = nile.data.load_pandas().data['volume']
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    endog1 = niledata.iloc[:20]
    endog2 = niledata.iloc[20:40]
    mod = sarimax.SARIMAX(endog1, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.smooth([0.5])
    assert_raises(ValueError, res1.append, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.extend, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.apply, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.append, endog2, fit_kwargs={'cov_kwds': {}})
    assert_raises(ValueError, res1.extend, endog2, fit_kwargs={'cov_kwds': {}})
    assert_raises(ValueError, res1.apply, endog2, fit_kwargs={'cov_kwds': {}})
    wrong_freq = niledata.iloc[20:40]
    wrong_freq.index = pd.date_range(start=niledata.index[0], periods=len(wrong_freq), freq='MS')
    message = 'Given `endog` does not have an index that extends the index of the model. Expected index frequency is'
    with pytest.raises(ValueError, match=message):
        res1.append(wrong_freq)
    with pytest.raises(ValueError, match=message):
        res1.extend(wrong_freq)
    message = 'Given `exog` does not have an index that extends the index of the model. Expected index frequency is'
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=wrong_freq)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=wrong_freq)
    not_cts = niledata.iloc[21:41]
    message = 'Given `endog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res1.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res1.extend(not_cts)
    message = 'Given `exog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=not_cts)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=not_cts)
    endog3 = pd.Series(niledata.iloc[:20].values)
    endog4 = pd.Series(niledata.iloc[:40].values)[20:]
    mod2 = sarimax.SARIMAX(endog3, order=(1, 0, 0), exog=endog3, concentrate_scale=True)
    res2 = mod2.smooth([0.2, 0.5])
    not_cts = pd.Series(niledata[:41].values)[21:]
    message = 'Given `endog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res2.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res2.extend(not_cts)
    message = 'Given `exog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res2.append(endog4, exog=not_cts)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res2.extend(endog4, exog=not_cts)