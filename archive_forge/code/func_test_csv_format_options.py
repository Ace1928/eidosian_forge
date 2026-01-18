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
def test_csv_format_options(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('skipped\ncol0\nfoo\nbar\n')
    dataset = ds.dataset(path, format='csv')
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'skipped': pa.array(['col0', 'foo', 'bar'])}))
    dataset = ds.dataset(path, format=ds.CsvFileFormat(read_options=pa.csv.ReadOptions(skip_rows=1)))
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'bar'])}))
    dataset = ds.dataset(path, format=ds.CsvFileFormat(read_options=pa.csv.ReadOptions(column_names=['foo'])))
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'foo': pa.array(['skipped', 'col0', 'foo', 'bar'])}))