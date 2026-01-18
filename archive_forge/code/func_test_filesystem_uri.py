import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def test_filesystem_uri(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    directory = tempdir / 'data_dir'
    directory.mkdir()
    path = directory / 'data.parquet'
    pq.write_table(table, str(path))
    result = pq.read_table(path, filesystem=fs.LocalFileSystem())
    assert result.equals(table)
    result = pq.read_table('data_dir/data.parquet', filesystem=util._filesystem_uri(tempdir))
    assert result.equals(table)