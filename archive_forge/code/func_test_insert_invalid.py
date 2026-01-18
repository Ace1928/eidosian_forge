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
def test_insert_invalid(self, data, invalid_scalar):
    item = invalid_scalar
    with pytest.raises((TypeError, ValueError)):
        data.insert(0, item)
    with pytest.raises((TypeError, ValueError)):
        data.insert(4, item)
    with pytest.raises((TypeError, ValueError)):
        data.insert(len(data) - 1, item)