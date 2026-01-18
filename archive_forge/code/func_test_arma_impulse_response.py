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
def test_arma_impulse_response():
    arrep = arma_impulse_response(armarep.ma, armarep.ar, leads=21)[1:]
    marep = arma_impulse_response(armarep.ar, armarep.ma, leads=21)[1:]
    assert_array_almost_equal(armarep.marep.ravel(), marep, 14)
    assert_array_almost_equal(-armarep.arrep.ravel(), arrep, 14)