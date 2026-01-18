from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_array_scalar_like_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)
    scalar_array = pd.array([scalar] * len(data), dtype=data.dtype)
    for scalar in [scalar, data.dtype.type(scalar)]:
        if is_bool_not_implemented(data, all_arithmetic_operators):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(data, scalar)
            with pytest.raises(NotImplementedError, match=msg):
                op(data, scalar_array)
        else:
            result = op(data, scalar)
            expected = op(data, scalar_array)
            tm.assert_extension_array_equal(result, expected)