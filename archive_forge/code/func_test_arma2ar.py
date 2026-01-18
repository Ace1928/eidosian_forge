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
def test_arma2ar(self):
    process1 = ArmaProcess.from_coeffs([], [0.8])
    vals = process1.arma2ar(100)
    assert_almost_equal(vals, (-0.8) ** np.arange(100.0))