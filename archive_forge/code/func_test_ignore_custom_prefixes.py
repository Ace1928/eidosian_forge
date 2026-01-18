import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def test_ignore_custom_prefixes(tempdir):
    part = ['xxx'] * 3 + ['yyy'] * 3
    table = pa.table([pa.array(range(len(part))), pa.array(part).dictionary_encode()], names=['index', '_part'])
    pq.write_to_dataset(table, str(tempdir), partition_cols=['_part'])
    private_duplicate = tempdir / '_private_duplicate'
    private_duplicate.mkdir()
    pq.write_to_dataset(table, str(private_duplicate), partition_cols=['_part'])
    read = pq.read_table(tempdir, ignore_prefixes=['_private'])
    assert read.equals(table)