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
def test_factorize_empty(self, data):
    codes, uniques = pd.factorize(data[:0])
    expected_codes = np.array([], dtype=np.intp)
    expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)
    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_extension_array_equal(uniques, expected_uniques)