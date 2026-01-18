import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.slow
@pytest.mark.large_memory
def test_metadata_exceeds_message_size():
    NCOLS = 1000
    NREPEATS = 4000
    table = pa.table({str(i): np.random.randn(10) for i in range(NCOLS)})
    with pa.BufferOutputStream() as out:
        pq.write_table(table, out)
        buf = out.getvalue()
    original_metadata = pq.read_metadata(pa.BufferReader(buf))
    metadata = pq.read_metadata(pa.BufferReader(buf))
    for i in range(NREPEATS):
        metadata.append_row_groups(original_metadata)
    with pa.BufferOutputStream() as out:
        metadata.write_metadata_file(out)
        buf = out.getvalue()
    metadata = pq.read_metadata(pa.BufferReader(buf))