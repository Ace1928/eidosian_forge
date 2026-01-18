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
def test_make_parquet_fragment_from_buffer(dataset_reader, pickle_module):
    arrays = [pa.array(['a', 'b', 'c']), pa.array([12, 11, 10]), pa.array(['dog', 'cat', 'rabbit'])]
    dictionary_arrays = [arrays[0].dictionary_encode(), arrays[1], arrays[2].dictionary_encode()]
    dictionary_format = ds.ParquetFileFormat(read_options=ds.ParquetReadOptions(dictionary_columns=['alpha', 'animal']), use_buffered_stream=True, buffer_size=4096)
    cases = [(arrays, ds.ParquetFileFormat()), (dictionary_arrays, dictionary_format)]
    for arrays, format_ in cases:
        table = pa.table(arrays, names=['alpha', 'num', 'animal'])
        out = pa.BufferOutputStream()
        pq.write_table(table, out)
        buffer = out.getvalue()
        fragment = format_.make_fragment(buffer)
        assert dataset_reader.to_table(fragment).equals(table)
        pickled = pickle_module.loads(pickle_module.dumps(fragment))
        assert dataset_reader.to_table(pickled).equals(table)