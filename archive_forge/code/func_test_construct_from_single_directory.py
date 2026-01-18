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
def test_construct_from_single_directory(tempdir, dataset_reader, pickle_module):
    directory = tempdir / 'single-directory'
    directory.mkdir()
    tables, paths = _create_directory_of_files(directory)
    d1 = ds.dataset(directory)
    d2 = ds.dataset(directory, filesystem=fs.LocalFileSystem())
    d3 = ds.dataset(directory.name, filesystem=_filesystem_uri(tempdir))
    t1 = dataset_reader.to_table(d1)
    t2 = dataset_reader.to_table(d2)
    t3 = dataset_reader.to_table(d3)
    assert t1 == t2 == t3
    for d in [d1, d2, d3]:
        restored = pickle_module.loads(pickle_module.dumps(d))
        assert dataset_reader.to_table(restored) == t1