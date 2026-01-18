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
def test_apply_simple_series(self, data):
    result = pd.Series(data).apply(id)
    assert isinstance(result, pd.Series)