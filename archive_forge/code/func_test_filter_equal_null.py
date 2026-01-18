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
def test_filter_equal_null(tempdir, dataset_reader):
    table = pa.table({'A': ['a', 'b', None]})
    _, path = _create_single_file(tempdir, table)
    dataset = ds.dataset(str(path))
    table = dataset_reader.to_table(dataset, filter=ds.field('A') == ds.scalar(None))
    assert table.num_rows == 0