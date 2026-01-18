from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_boolean_empty(self):
    ser = Series([], dtype=np.int64)
    ser.index.name = 'index_name'
    ser = ser[ser.isna()]
    assert ser.index.name == 'index_name'
    assert ser.dtype == np.int64
    ser = Series(['A', 'B'], dtype=object)
    expected = Series(dtype=object, index=Index([], dtype='int64'))
    result = ser[Series([], dtype=object)]
    tm.assert_series_equal(result, expected)
    msg = 'Unalignable boolean Series provided as indexer \\(index of the boolean Series and of the indexed object do not match'
    with pytest.raises(IndexingError, match=msg):
        ser[Series([], dtype=bool)]
    with pytest.raises(IndexingError, match=msg):
        ser[Series([True], dtype=bool)]