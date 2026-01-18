from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
@final
def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
    arr = ser.array
    df = pd.DataFrame({'a': arr})
    kwargs = {'ddof': 1} if op_name in ['var', 'std'] else {}
    cmp_dtype = self._get_expected_reduction_dtype(arr, op_name, skipna)
    result1 = arr._reduce(op_name, skipna=skipna, keepdims=True, **kwargs)
    result2 = getattr(df, op_name)(skipna=skipna, **kwargs).array
    tm.assert_extension_array_equal(result1, result2)
    if not skipna and ser.isna().any():
        expected = pd.array([pd.NA], dtype=cmp_dtype)
    else:
        exp_value = getattr(ser.dropna(), op_name)()
        expected = pd.array([exp_value], dtype=cmp_dtype)
    tm.assert_extension_array_equal(result1, expected)