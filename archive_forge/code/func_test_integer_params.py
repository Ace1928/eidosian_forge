from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_integer_params():
    mod = sarimax.SARIMAX([1, 1, 1], order=(1, 0, 0), exog=[2, 2, 2], concentrate_scale=True)
    res = mod.filter([1, 0])
    p = res.predict(end=5, dynamic=True, exog=[3, 3, 4])
    assert_equal(p.dtype, np.float64)