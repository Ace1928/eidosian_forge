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
def test_construct_from_list_of_files(tempdir, dataset_reader):
    directory = tempdir / 'list-of-files'
    directory.mkdir()
    tables, paths = _create_directory_of_files(directory)
    relative_paths = [p.relative_to(tempdir) for p in paths]
    with change_cwd(tempdir):
        d1 = ds.dataset(relative_paths)
        t1 = dataset_reader.to_table(d1)
        assert len(t1) == sum(map(len, tables))
    d2 = ds.dataset(relative_paths, filesystem=_filesystem_uri(tempdir))
    t2 = dataset_reader.to_table(d2)
    d3 = ds.dataset(paths)
    t3 = dataset_reader.to_table(d3)
    d4 = ds.dataset(paths, filesystem=fs.LocalFileSystem())
    t4 = dataset_reader.to_table(d4)
    assert t1 == t2 == t3 == t4