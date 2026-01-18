from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.parametrize('array_factory', [lambda: pa.array([0, None] * 10), lambda: pa.array([0, None] * 10).dictionary_encode(), lambda: pa.array(['', None] * 10), lambda: pa.array(['', None] * 10).dictionary_encode()])
@pytest.mark.parametrize('read_dictionary', [False, True])
def test_buffer_contents(array_factory, read_dictionary):
    orig_table = pa.Table.from_pydict({'col': array_factory()})
    bio = io.BytesIO()
    pq.write_table(orig_table, bio, use_dictionary=True)
    bio.seek(0)
    read_dictionary = ['col'] if read_dictionary else None
    table = pq.read_table(bio, use_threads=False, read_dictionary=read_dictionary)
    for col in table.columns:
        [chunk] = col.chunks
        buf = chunk.buffers()[1]
        assert buf.to_pybytes() == buf.size * b'\x00'