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
def test_debiased():
    mod = ExponentialSmoothing(housing_data, trend='add', initialization_method='estimated')
    res = mod.fit()
    res2 = mod.fit(remove_bias=True)
    assert np.any(res.fittedvalues != res2.fittedvalues)
    err2 = housing_data.iloc[:, 0] - res2.fittedvalues
    assert_almost_equal(err2.mean(), 0.0)
    assert isinstance(res.summary().as_text(), str)
    assert isinstance(res2.summary().as_text(), str)