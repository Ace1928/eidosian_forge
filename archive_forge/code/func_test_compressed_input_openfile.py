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
@pytest.mark.gzip
def test_compressed_input_openfile(tmpdir):
    if not Codec.is_available('gzip'):
        pytest.skip('gzip support is not built')
    data = b'some test data\n' * 10 + b'eof\n'
    fn = str(tmpdir / 'test_compressed_input_openfile.gz')
    with gzip.open(fn, 'wb') as f:
        f.write(data)
    with pa.CompressedInputStream(fn, 'gzip') as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert compressed.closed
    with pa.CompressedInputStream(pathlib.Path(fn), 'gzip') as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert compressed.closed
    f = open(fn, 'rb')
    with pa.CompressedInputStream(f, 'gzip') as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert f.closed