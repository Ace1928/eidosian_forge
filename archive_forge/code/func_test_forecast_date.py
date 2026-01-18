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
def test_forecast_date(data):
    res = ARDL(data.y, 3).fit()
    numeric = res.forecast(12)
    date = res.forecast('1990-07-01')
    assert_allclose(numeric, date)
    assert_index_equal(numeric.index, date.index)