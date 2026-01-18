from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('box', [pd.Series, pd.DataFrame, pd.Index])
@pytest.mark.parametrize('op_name', [x for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods if not x.startswith('__r')])
def test_direct_arith_with_ndframe_returns_not_implemented(self, data, box, op_name):
    other = box(data)
    if hasattr(data, op_name):
        result = getattr(data, op_name)(other)
        assert result is NotImplemented