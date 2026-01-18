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
def test_construct_from_single_file(tempdir, dataset_reader, pickle_module):
    directory = tempdir / 'single-file'
    directory.mkdir()
    table, path = _create_single_file(directory)
    relative_path = path.relative_to(directory)
    d1 = ds.dataset(path)
    d2 = ds.dataset(path, filesystem=fs.LocalFileSystem())
    d3 = ds.dataset(str(relative_path), filesystem=_filesystem_uri(directory))
    d4 = pickle_module.loads(pickle_module.dumps(d1))
    assert dataset_reader.to_table(d1) == dataset_reader.to_table(d2) == dataset_reader.to_table(d3) == dataset_reader.to_table(d4)