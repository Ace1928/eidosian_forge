from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_array_NA(data, all_arithmetic_operators):
    data, _ = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    scalar = pd.NA
    scalar_array = pd.array([pd.NA] * len(data), dtype=data.dtype)
    mask = data._mask.copy()
    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, scalar)
        tm.assert_numpy_array_equal(mask, data._mask)
        return
    result = op(data, scalar)
    tm.assert_numpy_array_equal(mask, data._mask)
    expected = op(data, scalar_array)
    tm.assert_numpy_array_equal(mask, data._mask)
    tm.assert_extension_array_equal(result, expected)