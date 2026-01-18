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
def test_generate_sample(self):
    process = ArmaProcess.from_coeffs([0.9])
    np.random.seed(12345)
    sample = process.generate_sample()
    np.random.seed(12345)
    expected = np.random.randn(100)
    for i in range(1, 100):
        expected[i] = 0.9 * expected[i - 1] + expected[i]
    assert_almost_equal(sample, expected)
    process = ArmaProcess.from_coeffs([1.6, -0.9])
    np.random.seed(12345)
    sample = process.generate_sample()
    np.random.seed(12345)
    expected = np.random.randn(100)
    expected[1] = 1.6 * expected[0] + expected[1]
    for i in range(2, 100):
        expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
    assert_almost_equal(sample, expected)
    process = ArmaProcess.from_coeffs([1.6, -0.9])
    np.random.seed(12345)
    sample = process.generate_sample(burnin=100)
    np.random.seed(12345)
    expected = np.random.randn(200)
    expected[1] = 1.6 * expected[0] + expected[1]
    for i in range(2, 200):
        expected[i] = 1.6 * expected[i - 1] - 0.9 * expected[i - 2] + expected[i]
    assert_almost_equal(sample, expected[100:])
    np.random.seed(12345)
    sample = process.generate_sample(nsample=(100, 5))
    assert_equal(sample.shape, (100, 5))