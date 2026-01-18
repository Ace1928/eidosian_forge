from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
@pytest.mark.parametrize('input_param,expected_error_message', [(None, 'Cannot deduce dtype from null object'), (1, 'data type not understood'), (1.2, 'data type not understood'), ('random string', 'data type ["\']random string["\'] not understood'), (pd.DataFrame([1, 2]), 'data type not understood')])
def test_get_dtype_fails(input_param, expected_error_message):
    expected_error_message += f"|Cannot interpret '{input_param}' as a data type"
    with pytest.raises(TypeError, match=expected_error_message):
        com._get_dtype(input_param)