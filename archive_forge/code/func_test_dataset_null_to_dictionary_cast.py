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
def test_dataset_null_to_dictionary_cast(tempdir, dataset_reader):
    table = pa.table({'a': [None, None]})
    pq.write_table(table, tempdir / 'test.parquet')
    schema = pa.schema([pa.field('a', pa.dictionary(pa.int32(), pa.string()))])
    fsds = ds.FileSystemDataset.from_paths(paths=[tempdir / 'test.parquet'], schema=schema, format=ds.ParquetFileFormat(), filesystem=fs.LocalFileSystem())
    table = dataset_reader.to_table(fsds)
    assert table.schema == schema