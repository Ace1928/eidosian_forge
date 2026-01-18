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
@pytest.mark.s3
def test_make_fragment_with_size(s3_example_simple):
    """
    Test passing file_size to make_fragment. Not all FS implementations make use
    of the file size (by implementing an OpenInputFile that takes a FileInfo), but
    s3 does, which is why it's used here.
    """
    table, path, fs, uri, host, port, access_key, secret_key = s3_example_simple
    file_format = ds.ParquetFileFormat()
    paths = [path]
    fragments = [file_format.make_fragment(path, fs) for path in paths]
    dataset = ds.FileSystemDataset(fragments, format=file_format, schema=table.schema, filesystem=fs)
    tbl = dataset.to_table()
    assert tbl.equals(table)
    sizes_true = [dataset.filesystem.get_file_info(x).size for x in dataset.files]
    fragments_with_size = [file_format.make_fragment(path, fs, file_size=size) for path, size in zip(paths, sizes_true)]
    dataset_with_size = ds.FileSystemDataset(fragments_with_size, format=file_format, schema=table.schema, filesystem=fs)
    tbl = dataset.to_table()
    assert tbl.equals(table)
    sizes_toosmall = [1 for path in paths]
    fragments_with_size = [file_format.make_fragment(path, fs, file_size=size) for path, size in zip(paths, sizes_toosmall)]
    dataset_with_size = ds.FileSystemDataset(fragments_with_size, format=file_format, schema=table.schema, filesystem=fs)
    with pytest.raises(pyarrow.lib.ArrowInvalid, match='Parquet file size is 1 bytes'):
        table = dataset_with_size.to_table()
    sizes_toolarge = [1000000 for path in paths]
    fragments_with_size = [file_format.make_fragment(path, fs, file_size=size) for path, size in zip(paths, sizes_toolarge)]
    dataset_with_size = ds.FileSystemDataset(fragments_with_size, format=file_format, schema=table.schema, filesystem=fs)
    with pytest.raises(OSError, match='HTTP status 416'):
        table = dataset_with_size.to_table()