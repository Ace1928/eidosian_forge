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
def test_arma_acf_compare_R_ARMAacf():
    bd_example_3_3_2 = arma_acf([1, -1, 0.25], [1, 1])
    assert_allclose(bd_example_3_3_2, results_arma_acf.bd_example_3_3_2)
    example_1 = arma_acf([1, -1, 0.25], [1, 1, 0.2])
    assert_allclose(example_1, results_arma_acf.custom_example_1)
    example_2 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3])
    assert_allclose(example_2, results_arma_acf.custom_example_2)
    example_3 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35])
    assert_allclose(example_3, results_arma_acf.custom_example_3)
    example_4 = arma_acf([1, -1, 0.25], [1, 1, 0.2, 0.3, -0.35, -0.1])
    assert_allclose(example_4, results_arma_acf.custom_example_4)
    example_5 = arma_acf([1, -1, 0.25, -0.1], [1, 1, 0.2])
    assert_allclose(example_5, results_arma_acf.custom_example_5)
    example_6 = arma_acf([1, -1, 0.25, -0.1, 0.05], [1, 1, 0.2])
    assert_allclose(example_6, results_arma_acf.custom_example_6)
    example_7 = arma_acf([1, -1, 0.25, -0.1, 0.05, -0.02], [1, 1, 0.2])
    assert_allclose(example_7, results_arma_acf.custom_example_7)