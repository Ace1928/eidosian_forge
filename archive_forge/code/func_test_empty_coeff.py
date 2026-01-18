from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
def test_empty_coeff(self):
    process = ArmaProcess()
    assert_equal(process.arcoefs, np.array([]))
    assert_equal(process.macoefs, np.array([]))
    process = ArmaProcess([1, -0.8])
    assert_equal(process.arcoefs, np.array([0.8]))
    assert_equal(process.macoefs, np.array([]))
    process = ArmaProcess(ma=[1, -0.8])
    assert_equal(process.arcoefs, np.array([]))
    assert_equal(process.macoefs, np.array([-0.8]))