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
@pytest.mark.parametrize('start', [None, 0, 2, 4])
@pytest.mark.parametrize('end', [None, 20])
@pytest.mark.parametrize('dynamic', [20, True])
def test_against_autoreg_predict_start_end(data, trend, seasonal, start, end, dynamic):
    ar = AutoReg(data.y, 3, trend=trend, seasonal=seasonal)
    ardl = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    ar_res = ar.fit()
    ardl_res = ardl.fit()
    ar_fcast = ar_res.predict(start=start, end=end, dynamic=dynamic)
    ardl_fcast = ardl_res.predict(start=start, end=end, dynamic=dynamic)
    assert_index_equal(ar_fcast.index, ardl_fcast.index)
    assert_allclose(ar_fcast, ardl_fcast)