from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('dtype', ['int64', 'int64[pyarrow]'])
def test_merge_arrow_and_numpy_dtypes(dtype):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2]}, dtype=dtype)
    df2 = DataFrame({'a': [1, 2]}, dtype='int64[pyarrow]')
    result = df.merge(df2)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)
    result = df2.merge(df)
    expected = df2.copy()
    tm.assert_frame_equal(result, expected)