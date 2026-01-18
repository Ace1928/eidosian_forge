from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
@pytest.fixture()
def bivariate_var_data(reset_randomstate):
    """A bivariate dataset for VAR estimation"""
    e = np.random.standard_normal((252, 2))
    y = np.zeros_like(e)
    y[:2] = e[:2]
    for i in range(2, 252):
        y[i] = 0.2 * y[i - 1] + 0.1 * y[i - 2] + e[i]
    return y