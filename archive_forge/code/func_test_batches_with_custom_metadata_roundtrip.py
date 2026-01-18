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
@pytest.mark.parametrize('ipc_type', ['stream', 'file'])
def test_batches_with_custom_metadata_roundtrip(ipc_type):
    df = pd.DataFrame({'foo': [1.5]})
    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()
    batch_count = 2
    file_factory = {'stream': pa.ipc.new_stream, 'file': pa.ipc.new_file}[ipc_type]
    with file_factory(sink, batch.schema) as writer:
        for i in range(batch_count):
            writer.write_batch(batch, custom_metadata={'batch_id': str(i)})
        writer.write_batch(batch)
    buffer = sink.getvalue()
    if ipc_type == 'stream':
        with pa.ipc.open_stream(buffer) as reader:
            batch_with_metas = list(reader.iter_batches_with_custom_metadata())
    else:
        with pa.ipc.open_file(buffer) as reader:
            batch_with_metas = [reader.get_batch_with_custom_metadata(i) for i in range(reader.num_record_batches)]
    for i in range(batch_count):
        assert batch_with_metas[i].batch.num_rows == 1
        assert isinstance(batch_with_metas[i].custom_metadata, pa.KeyValueMetadata)
        assert batch_with_metas[i].custom_metadata == {'batch_id': str(i)}
    assert batch_with_metas[batch_count].batch.num_rows == 1
    assert batch_with_metas[batch_count].custom_metadata is None