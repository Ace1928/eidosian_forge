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
def test_ipc_format(tempdir, dataset_reader):
    table = pa.table({'a': pa.array([1, 2, 3], type='int8'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    path = str(tempdir / 'test.arrow')
    with pa.output_stream(path) as sink:
        writer = pa.RecordBatchFileWriter(sink, table.schema)
        writer.write_batch(table.to_batches()[0])
        writer.close()
    dataset = ds.dataset(path, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)
    assert_dataset_fragment_convenience_methods(dataset)
    for format_str in ['ipc', 'arrow']:
        dataset = ds.dataset(path, format=format_str)
        result = dataset_reader.to_table(dataset)
        assert result.equals(table)