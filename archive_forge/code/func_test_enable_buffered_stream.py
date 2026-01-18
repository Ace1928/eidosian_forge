from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.pandas
def test_enable_buffered_stream(tempdir):
    df = alltypes_sample(size=10)
    table = pa.Table.from_pandas(df)
    _check_roundtrip(table, read_table_kwargs={'buffer_size': 1025}, version='2.6')
    filename = str(tempdir / 'tmp_file')
    with open(filename, 'wb') as f:
        _write_table(table, f, version='2.6')
    table_read = pq.read_pandas(filename, buffer_size=4096)
    assert table_read.equals(table)