from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.parametrize(['test_data', 'kind'], [(['foo', 'bar'], 21), ([1.5, 2.5, 3.5], 2), ([1, 2, 3, 4], 0)])
def test_array_to_pyarrowcolumn(test_data, kind):
    arr = pa.array(test_data)
    arr_column = _PyArrowColumn(arr)
    assert arr_column._col == arr
    assert arr_column.size() == len(test_data)
    assert arr_column.dtype[0] == kind
    assert arr_column.num_chunks() == 1
    assert arr_column.null_count == 0
    assert arr_column.get_buffers()['validity'] is None
    assert len(list(arr_column.get_chunks())) == 1
    for chunk in arr_column.get_chunks():
        assert chunk == arr_column