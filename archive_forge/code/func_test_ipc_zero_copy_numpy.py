from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_ipc_zero_copy_numpy():
    df = pd.DataFrame({'foo': [1.5]})
    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()
    write_file(batch, sink)
    buffer = sink.getvalue()
    reader = pa.BufferReader(buffer)
    batches = read_file(reader)
    data = batches[0].to_pandas()
    rdf = pd.DataFrame(data)
    assert_frame_equal(df, rdf)