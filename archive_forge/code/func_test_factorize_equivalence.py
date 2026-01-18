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
def test_factorize_equivalence(self, data_for_grouping):
    codes_1, uniques_1 = pd.factorize(data_for_grouping, use_na_sentinel=True)
    codes_2, uniques_2 = data_for_grouping.factorize(use_na_sentinel=True)
    tm.assert_numpy_array_equal(codes_1, codes_2)
    tm.assert_extension_array_equal(uniques_1, uniques_2)
    assert len(uniques_1) == len(pd.unique(uniques_1))
    assert uniques_1.dtype == data_for_grouping.dtype