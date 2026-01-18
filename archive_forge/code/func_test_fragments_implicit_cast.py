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
def test_fragments_implicit_cast(tempdir):
    table = pa.table([range(8), [1] * 4 + [2] * 4], names=['col', 'part'])
    path = str(tempdir / 'test_parquet_dataset')
    pq.write_to_dataset(table, path, partition_cols=['part'])
    part = ds.partitioning(pa.schema([('part', 'int8')]), flavor='hive')
    dataset = ds.dataset(path, format='parquet', partitioning=part)
    fragments = dataset.get_fragments(filter=ds.field('part') >= 2)
    assert len(list(fragments)) == 1