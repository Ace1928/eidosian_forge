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
@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_repr(tempdir, dataset):
    fragment = list(dataset.get_fragments())[0]
    assert repr(fragment) == '<pyarrow.dataset.ParquetFileFragment path=subdir/1/xxx/file0.parquet partition=[key=xxx, group=1]>' or repr(fragment) == '<pyarrow.dataset.ParquetFileFragment path=subdir/1/xxx/file0.parquet partition=[group=1, key=xxx]>'
    table, path = _create_single_file(tempdir)
    dataset = ds.dataset(path, format='parquet')
    fragment = list(dataset.get_fragments())[0]
    assert repr(fragment) == '<pyarrow.dataset.ParquetFileFragment path={}>'.format(dataset.filesystem.normalize_path(str(path)))
    path = tempdir / 'data.feather'
    pa.feather.write_feather(table, path)
    dataset = ds.dataset(path, format='feather')
    fragment = list(dataset.get_fragments())[0]
    assert repr(fragment) == '<pyarrow.dataset.FileFragment type=ipc path={}>'.format(dataset.filesystem.normalize_path(str(path)))