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
def test_make_csv_fragment_from_buffer(dataset_reader, pickle_module):
    content = textwrap.dedent('\n        alpha,num,animal\n        a,12,dog\n        b,11,cat\n        c,10,rabbit\n    ')
    buffer = pa.py_buffer(content.encode('utf-8'))
    csv_format = ds.CsvFileFormat()
    fragment = csv_format.make_fragment(buffer)
    assert isinstance(fragment.open(), pa.BufferReader)
    expected = pa.table([['a', 'b', 'c'], [12, 11, 10], ['dog', 'cat', 'rabbit']], names=['alpha', 'num', 'animal'])
    assert dataset_reader.to_table(fragment).equals(expected)
    pickled = pickle_module.loads(pickle_module.dumps(fragment))
    assert dataset_reader.to_table(pickled).equals(fragment.to_table())