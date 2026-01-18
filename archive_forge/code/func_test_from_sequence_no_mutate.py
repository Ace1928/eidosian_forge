import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('copy', [True, False])
def test_from_sequence_no_mutate(copy, cls, dtype):
    nan_arr = np.array(['a', np.nan], dtype=object)
    expected_input = nan_arr.copy()
    na_arr = np.array(['a', pd.NA], dtype=object)
    result = cls._from_sequence(nan_arr, dtype=dtype, copy=copy)
    if cls in (ArrowStringArray, ArrowStringArrayNumpySemantics):
        import pyarrow as pa
        expected = cls(pa.array(na_arr, type=pa.string(), from_pandas=True))
    else:
        expected = cls(na_arr)
    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(nan_arr, expected_input)