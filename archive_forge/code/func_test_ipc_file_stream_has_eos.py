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
def test_ipc_file_stream_has_eos():
    df = pd.DataFrame({'foo': [1.5]})
    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()
    write_file(batch, sink)
    buffer = sink.getvalue()
    reader = pa.ipc.open_stream(buffer[8:])
    rdf = reader.read_pandas()
    assert_frame_equal(df, rdf)