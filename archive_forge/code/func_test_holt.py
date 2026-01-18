from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_holt(self):
    fit1 = Holt(self.air_ausair, initialization_method='legacy-heuristic').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
    fit2 = Holt(self.air_ausair, exponential=True, initialization_method='legacy-heuristic').fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
    fit3 = Holt(self.air_ausair, damped_trend=True, initialization_method='estimated').fit(smoothing_level=0.8, smoothing_trend=0.2)
    assert_almost_equal(fit1.forecast(5), [43.76, 45.59, 47.43, 49.27, 51.1], 2)
    assert_almost_equal(fit1.trend, [3.617628, 3.59006512, 3.33438212, 3.23657639, 2.69263502, 2.46388914, 2.2229097, 1.95959226, 1.47054601, 1.3604894, 1.28045881, 1.20355193, 1.88267152, 2.09564416, 1.83655482], 4)
    assert_almost_equal(fit1.fittedfcast, [21.8601, 22.032368, 25.48461872, 27.54058587, 30.28813356, 30.26106173, 31.58122149, 32.599234, 33.24223906, 32.26755382, 33.07776017, 33.95806605, 34.77708354, 40.05535303, 43.21586036, 43.75696849], 4)
    assert_almost_equal(fit2.forecast(5), [44.6, 47.24, 50.04, 53.01, 56.15], 2)
    assert_almost_equal(fit3.forecast(5), [42.85, 43.81, 44.66, 45.41, 46.06], 2)