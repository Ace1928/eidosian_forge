import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def test_argsort_missing(self, data_missing_for_sorting):
    msg = 'The behavior of Series.argsort in the presence of NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.Series(data_missing_for_sorting).argsort()
    expected = pd.Series(np.array([1, -1, 0], dtype=np.intp))
    tm.assert_series_equal(result, expected)