import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_aggregate_directory(reduction_func):
    if reduction_func in ['corrwith', 'nth']:
        return None
    obj = DataFrame([[0, 1], [0, np.nan]])
    result_reduced_series = obj.groupby(0).agg(reduction_func)
    result_reduced_frame = obj.groupby(0).agg({1: reduction_func})
    if reduction_func in ['size', 'ngroup']:
        tm.assert_series_equal(result_reduced_series, result_reduced_frame[1], check_names=False)
    else:
        tm.assert_frame_equal(result_reduced_series, result_reduced_frame)
        tm.assert_series_equal(result_reduced_series.dtypes, result_reduced_frame.dtypes)