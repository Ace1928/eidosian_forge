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
def test_filter_compute_expression(tempdir, dataset_reader):
    table = pa.table({'A': ['a', 'b', None, 'a', 'c'], 'B': [datetime.datetime(2022, 1, 1, i) for i in range(5)], 'C': [datetime.datetime(2022, 1, i) for i in range(1, 6)]})
    _, path = _create_single_file(tempdir, table)
    dataset = ds.dataset(str(path))
    filter_ = pc.is_in(ds.field('A'), pa.array(['a', 'b']))
    assert dataset_reader.to_table(dataset, filter=filter_).num_rows == 3
    filter_ = pc.hour(ds.field('B')) >= 3
    assert dataset_reader.to_table(dataset, filter=filter_).num_rows == 2
    days = pc.days_between(ds.field('B'), ds.field('C'))
    result = dataset_reader.to_table(dataset, columns={'days': days})
    assert result['days'].to_pylist() == [0, 1, 2, 3, 4]