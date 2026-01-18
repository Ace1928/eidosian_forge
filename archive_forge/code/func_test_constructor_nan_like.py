import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('na', [np.nan, np.float64('nan'), float('nan'), None, pd.NA])
def test_constructor_nan_like(na):
    expected = pd.arrays.StringArray(np.array(['a', pd.NA]))
    tm.assert_extension_array_equal(pd.arrays.StringArray(np.array(['a', na], dtype='object')), expected)