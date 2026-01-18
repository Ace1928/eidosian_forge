from statsmodels.compat.pandas import PD_LT_2_2_0
from datetime import datetime
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
def test_get_predict_start_end():
    index = pd.date_range(start='1970-01-01', end='1990-01-01', freq='YS')
    endog = pd.Series(np.zeros(10), index[:10])
    model = TimeSeriesModel(endog)
    predict_starts = [1, '1971-01-01', datetime(1971, 1, 1), index[1]]
    predict_ends = [20, '1990-01-01', datetime(1990, 1, 1), index[-1]]
    desired = (1, 9, 11)
    for start in predict_starts:
        for end in predict_ends:
            assert_equal(model._get_prediction_index(start, end)[:3], desired)