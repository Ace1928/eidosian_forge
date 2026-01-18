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
def test_multithreaded_read():
    df = alltypes_sample(size=10000)
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    _write_table(table, buf, compression='SNAPPY', version='2.6')
    buf.seek(0)
    table1 = _read_table(buf, use_threads=True)
    buf.seek(0)
    table2 = _read_table(buf, use_threads=False)
    assert table1.equals(table2)