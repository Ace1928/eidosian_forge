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
def test_basin_hopping(reset_randomstate):
    mod = ExponentialSmoothing(housing_data, trend='add', initialization_method='estimated')
    res = mod.fit()
    res2 = mod.fit(method='basinhopping')
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res2.summary().as_text(), str)
    tol = 1e-05
    assert res2.sse <= res.sse + tol
    res3 = mod.fit(method='basinhopping')
    assert_almost_equal(res2.sse, res3.sse, decimal=2)