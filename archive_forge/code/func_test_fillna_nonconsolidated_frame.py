import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_nonconsolidated_frame():
    df = DataFrame([[1, 1, 1, 1.0], [2, 2, 2, 2.0], [3, 3, 3, 3.0]], columns=['i1', 'i2', 'i3', 'f1'])
    df_nonconsol = df.pivot(index='i1', columns='i2')
    result = df_nonconsol.fillna(0)
    assert result.isna().sum().sum() == 0