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
def test_incompatible_schema_hang(tempdir, dataset_reader):
    fn = tempdir / 'data.parquet'
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, fn)
    schema = pa.schema([('a', pa.null())])
    dataset = ds.dataset([str(fn)] * 100, schema=schema)
    assert dataset.schema.equals(schema)
    scanner = dataset_reader.scanner(dataset)
    with pytest.raises(NotImplementedError, match='Unsupported cast from int64 to null'):
        reader = scanner.to_reader()
        reader.read_all()