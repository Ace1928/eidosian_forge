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
def test_arma_acov_compare_theoretical_arma_acov():

    def arma_acovf_historical(ar, ma, nobs=10):
        if np.abs(np.sum(ar) - 1) > 0.9:
            nobs_ir = max(1000, 2 * nobs)
        else:
            nobs_ir = max(100, 2 * nobs)
        ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        while ir[-1] > 5 * 1e-05:
            nobs_ir *= 10
            ir = arma_impulse_response(ar, ma, leads=nobs_ir)
        if nobs_ir > 50000 and nobs < 1001:
            end = len(ir)
            acovf = np.array([np.dot(ir[:end - nobs - t], ir[t:end - nobs]) for t in range(nobs)])
        else:
            acovf = np.correlate(ir, ir, 'full')[len(ir) - 1:]
        return acovf[:nobs]
    assert_allclose(arma_acovf([1, -0.5], [1, 0.2]), arma_acovf_historical([1, -0.5], [1, 0.2]))
    assert_allclose(arma_acovf([1, -0.99], [1, 0.2]), arma_acovf_historical([1, -0.99], [1, 0.2]))