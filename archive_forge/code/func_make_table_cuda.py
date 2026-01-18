import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def make_table_cuda():
    htable = make_table()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, htable.schema) as out:
        out.write_table(htable)
    hbuf = pa.py_buffer(sink.getvalue().to_pybytes())
    dbuf = global_context.new_buffer(len(hbuf))
    dbuf.copy_from_host(hbuf, nbytes=len(hbuf))
    dtable = pa.ipc.open_stream(cuda.BufferReader(dbuf)).read_all()
    return (hbuf, htable, dbuf, dtable)