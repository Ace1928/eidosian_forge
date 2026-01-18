from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_to_records(self):
    arr = np.zeros((2,), dtype='i4,f4,S10')
    arr[:] = [(1, 2.0, 'Hello'), (2, 3.0, 'World')]
    DataFrame.from_records(arr)
    index = Index(np.arange(len(arr))[::-1])
    indexed_frame = DataFrame.from_records(arr, index=index)
    tm.assert_index_equal(indexed_frame.index, index)
    arr2 = np.zeros((2, 3))
    tm.assert_frame_equal(DataFrame.from_records(arr2), DataFrame(arr2))
    msg = '|'.join(['Length of values \\(2\\) does not match length of index \\(1\\)'])
    with pytest.raises(ValueError, match=msg):
        DataFrame.from_records(arr, index=index[:-1])
    indexed_frame = DataFrame.from_records(arr, index='f1')
    records = indexed_frame.to_records()
    assert len(records.dtype.names) == 3
    records = indexed_frame.to_records(index=False)
    assert len(records.dtype.names) == 2
    assert 'index' not in records.dtype.names