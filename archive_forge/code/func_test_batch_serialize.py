import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_batch_serialize():
    batch = make_recordbatch(10)
    hbuf = batch.serialize()
    cbuf = cuda.serialize_record_batch(batch, global_context)
    cbatch = cuda.read_record_batch(cbuf, batch.schema)
    assert isinstance(cbatch, pa.RecordBatch)
    assert batch.schema == cbatch.schema
    assert batch.num_columns == cbatch.num_columns
    assert batch.num_rows == cbatch.num_rows
    buf = cbuf.copy_to_host()
    assert hbuf.equals(buf)
    batch2 = pa.ipc.read_record_batch(buf, batch.schema)
    assert hbuf.equals(batch2.serialize())
    assert batch.num_columns == batch2.num_columns
    assert batch.num_rows == batch2.num_rows
    assert batch.column(0).equals(batch2.column(0))
    assert batch.equals(batch2)