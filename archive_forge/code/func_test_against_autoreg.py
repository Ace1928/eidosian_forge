from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
def test_against_autoreg(data, trend, seasonal):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()
    assert_allclose(ar_res.params, ardl_res.params)
    assert ar_res.ar_lags == ardl_res.ar_lags
    assert ar.trend == ardl.trend
    assert ar.seasonal == ardl.seasonal
    ar_fcast = ar_res.forecast(12)
    ardl_fcast = ardl_res.forecast(12)
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)
    ar_fcast = ar_res.predict()
    ardl_fcast = ardl_res.predict()
    assert_allclose(ar_fcast, ardl_fcast)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)