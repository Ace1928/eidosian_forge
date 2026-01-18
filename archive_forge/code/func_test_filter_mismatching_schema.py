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
def test_filter_mismatching_schema(tempdir, dataset_reader):
    table = pa.table({'col': pa.array([1, 2, 3, 4], type='int32')})
    pq.write_table(table, str(tempdir / 'data.parquet'))
    schema = pa.schema([('col', pa.int64())])
    dataset = ds.dataset(tempdir / 'data.parquet', format='parquet', schema=schema)
    filtered = dataset_reader.to_table(dataset, filter=ds.field('col') > 2)
    assert filtered['col'].equals(table['col'].cast('int64').slice(2))
    fragment = list(dataset.get_fragments())[0]
    filtered = dataset_reader.to_table(fragment, filter=ds.field('col') > 2, schema=schema)
    assert filtered['col'].equals(table['col'].cast('int64').slice(2))