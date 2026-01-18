import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
def test_file_format_inspect_fsspec(tempdir):
    fsspec = pytest.importorskip('fsspec')
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'
    pq.write_table(table, path)
    fsspec_fs = fsspec.filesystem('file')
    assert fsspec_fs.ls(tempdir)[0].endswith('data.parquet')
    format = ds.ParquetFileFormat()
    filesystem = fs.PyFileSystem(fs.FSSpecHandler(fsspec_fs))
    schema = format.inspect(path, filesystem)
    assert schema.equals(table.schema)
    fragment = format.make_fragment(path, filesystem)
    assert fragment.physical_schema.equals(table.schema)