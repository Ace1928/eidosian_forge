import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.large_memory
def test_large_binary_overflow():
    s = b'x' * (1 << 31)
    arr = pa.array([s], type=pa.large_binary())
    table = pa.Table.from_arrays([arr], names=['strs'])
    for use_dictionary in [False, True]:
        writer = pa.BufferOutputStream()
        with pytest.raises(pa.ArrowInvalid, match='Parquet cannot store strings with size 2GB or more'):
            _write_table(table, writer, use_dictionary=use_dictionary)