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
def test_hash_pandas_object_works(self, data, as_frame):
    data = pd.Series(data)
    if as_frame:
        data = data.to_frame()
    a = pd.util.hash_pandas_object(data)
    b = pd.util.hash_pandas_object(data)
    tm.assert_equal(a, b)