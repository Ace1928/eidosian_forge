import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.slow
@pytest.mark.large_memory
def test_large_table_int32_overflow():
    size = np.iinfo('int32').max + 1
    arr = np.ones(size, dtype='uint8')
    parr = pa.array(arr, type=pa.uint8())
    table = pa.Table.from_arrays([parr], names=['one'])
    f = io.BytesIO()
    _write_table(table, f)