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
def test_filter_timestamp(tempdir, dataset_reader):
    path = tempdir / 'test_partition_timestamps'
    table = pa.table({'dates': ['2012-01-01', '2012-01-02'] * 5, 'id': range(10)})
    part = ds.partitioning(table.select(['dates']).schema, flavor='hive')
    ds.write_dataset(table, path, partitioning=part, format='feather')
    part = ds.partitioning(pa.schema([('dates', pa.timestamp('s'))]), flavor='hive')
    dataset = ds.dataset(path, format='feather', partitioning=part)
    condition = ds.field('dates') > pd.Timestamp('2012-01-01')
    table = dataset_reader.to_table(dataset, filter=condition)
    assert table.column('id').to_pylist() == [1, 3, 5, 7, 9]
    import datetime
    condition = ds.field('dates') > datetime.datetime(2012, 1, 1)
    table = dataset_reader.to_table(dataset, filter=condition)
    assert table.column('id').to_pylist() == [1, 3, 5, 7, 9]