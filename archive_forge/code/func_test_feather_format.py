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
def test_feather_format(tempdir, dataset_reader):
    from pyarrow.feather import write_feather
    table = pa.table({'a': pa.array([1, 2, 3], type='int8'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    basedir = tempdir / 'feather_dataset'
    basedir.mkdir()
    write_feather(table, str(basedir / 'data.feather'))
    dataset = ds.dataset(basedir, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)
    assert_dataset_fragment_convenience_methods(dataset)
    dataset = ds.dataset(basedir, format='feather')
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)
    result = dataset_reader.to_table(dataset, columns=['b', 'a'])
    assert result.column_names == ['b', 'a']
    result = dataset_reader.to_table(dataset, columns=['a', 'a'])
    assert result.column_names == ['a', 'a']
    write_feather(table, str(basedir / 'data1.feather'), version=1)
    with pytest.raises(ValueError):
        dataset_reader.to_table(ds.dataset(basedir, format='feather'))