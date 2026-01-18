import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
@pytest.fixture
def data_for_grouping(dtype):
    b = True
    a = False
    na = np.nan
    return pd.array([b, b, na, na, a, a, b], dtype=dtype)