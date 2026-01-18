import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_mock_output_stream():
    val = b'dataabcdef'
    f1 = pa.MockOutputStream()
    f2 = pa.BufferOutputStream()
    K = 1000
    for i in range(K):
        f1.write(val)
        f2.write(val)
    assert f1.size() == len(f2.getvalue())
    record_batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ['a'])
    f1 = pa.MockOutputStream()
    f2 = pa.BufferOutputStream()
    stream_writer1 = pa.RecordBatchStreamWriter(f1, record_batch.schema)
    stream_writer2 = pa.RecordBatchStreamWriter(f2, record_batch.schema)
    stream_writer1.write_batch(record_batch)
    stream_writer2.write_batch(record_batch)
    stream_writer1.close()
    stream_writer2.close()
    assert f1.size() == len(f2.getvalue())