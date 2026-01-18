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
def test_from_roots(self):
    ar = [1.8, -0.9]
    ma = [0.3]
    ar.insert(0, -1)
    ma.insert(0, 1)
    ar_p = -1 * np.array(ar)
    ma_p = ma
    process_direct = ArmaProcess(ar_p, ma_p)
    process = ArmaProcess.from_roots(np.array(process_direct.maroots), np.array(process_direct.arroots))
    assert_almost_equal(process.arcoefs, process_direct.arcoefs)
    assert_almost_equal(process.macoefs, process_direct.macoefs)
    assert_almost_equal(process.nobs, process_direct.nobs)
    assert_almost_equal(process.maroots, process_direct.maroots)
    assert_almost_equal(process.arroots, process_direct.arroots)
    assert_almost_equal(process.isinvertible, process_direct.isinvertible)
    assert_almost_equal(process.isstationary, process_direct.isstationary)
    process_direct = ArmaProcess(ar=ar_p)
    process = ArmaProcess.from_roots(arroots=np.array(process_direct.arroots))
    assert_almost_equal(process.arcoefs, process_direct.arcoefs)
    assert_almost_equal(process.macoefs, process_direct.macoefs)
    assert_almost_equal(process.nobs, process_direct.nobs)
    assert_almost_equal(process.maroots, process_direct.maroots)
    assert_almost_equal(process.arroots, process_direct.arroots)
    assert_almost_equal(process.isinvertible, process_direct.isinvertible)
    assert_almost_equal(process.isstationary, process_direct.isstationary)
    process_direct = ArmaProcess(ma=ma_p)
    process = ArmaProcess.from_roots(maroots=np.array(process_direct.maroots))
    assert_almost_equal(process.arcoefs, process_direct.arcoefs)
    assert_almost_equal(process.macoefs, process_direct.macoefs)
    assert_almost_equal(process.nobs, process_direct.nobs)
    assert_almost_equal(process.maroots, process_direct.maroots)
    assert_almost_equal(process.arroots, process_direct.arroots)
    assert_almost_equal(process.isinvertible, process_direct.isinvertible)
    assert_almost_equal(process.isstationary, process_direct.isstationary)
    process_direct = ArmaProcess()
    process = ArmaProcess.from_roots()
    assert_almost_equal(process.arcoefs, process_direct.arcoefs)
    assert_almost_equal(process.macoefs, process_direct.macoefs)
    assert_almost_equal(process.nobs, process_direct.nobs)
    assert_almost_equal(process.maroots, process_direct.maroots)
    assert_almost_equal(process.arroots, process_direct.arroots)
    assert_almost_equal(process.isinvertible, process_direct.isinvertible)
    assert_almost_equal(process.isstationary, process_direct.isstationary)