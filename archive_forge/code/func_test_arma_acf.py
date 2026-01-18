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
def test_arma_acf():
    N = 20
    phi = 0.9
    sigma = 1
    rep1 = arma_acf([1, -phi], [1], N)
    acovf = np.array([1.0 * sigma * phi ** i / (1 - phi ** 2) for i in range(N)])
    rep2 = acovf / (1.0 / (1 - phi ** 2))
    assert_allclose(rep1, rep2)