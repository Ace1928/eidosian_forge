import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.large_memory
@pytest.mark.pandas
def test_chunked_binary_error_message():
    values = [b'x'] + [b'x' * (1 << 20)] * 2 * (1 << 10)
    df = pd.DataFrame({'byte_col': values})
    buf = io.BytesIO()
    write_feather(df, buf, version=2)
    result = read_feather(pa.BufferReader(buf.getvalue()))
    assert_frame_equal(result, df)
    with pytest.raises(ValueError, match="'byte_col' exceeds 2GB maximum capacity of a Feather binary column. This restriction may be lifted in the future"):
        write_feather(df, io.BytesIO(), version=1)