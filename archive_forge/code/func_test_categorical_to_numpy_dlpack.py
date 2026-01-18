from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
@pytest.mark.skipif(np_version_lt1p23, reason='Numpy > 1.23 required')
def test_categorical_to_numpy_dlpack():
    df = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'a'])})
    col = df.__dataframe__().get_column_by_name('A')
    result = np.from_dlpack(col.get_buffers()['data'][0])
    expected = np.array([0, 1, 0], dtype='int8')
    tm.assert_numpy_array_equal(result, expected)