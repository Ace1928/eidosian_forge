from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
    if all_arithmetic_operators == '__rmod__' and is_string_dtype(data.dtype):
        pytest.skip('Skip testing Python string formatting')
    op_name = all_arithmetic_operators
    ser = pd.Series(data)
    self.check_opname(ser, op_name, ser.iloc[0])