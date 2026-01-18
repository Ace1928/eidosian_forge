from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_get_index_loc_quarterly():
    ix = pd.date_range('2000Q1', periods=8, freq='QS')
    endog = pd.Series(np.zeros(8), index=ix)
    mod = tsa_model.TimeSeriesModel(endog)
    loc, index, _ = mod._get_index_loc('2003Q2')
    assert_equal(index[loc], pd.Timestamp('2003Q2'))