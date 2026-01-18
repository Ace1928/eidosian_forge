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
def test_buffered_output_stream():
    np_buf = np.zeros(100, dtype=np.int8)
    buf = pa.py_buffer(np_buf)
    raw = pa.FixedSizeBufferWriter(buf)
    f = pa.BufferedOutputStream(raw, buffer_size=4)
    f.write(b'12')
    assert np_buf[:4].tobytes() == b'\x00\x00\x00\x00'
    f.flush()
    assert np_buf[:4].tobytes() == b'12\x00\x00'
    f.write(b'3456789')
    f.close()
    assert f.closed
    assert raw.closed
    assert np_buf[:10].tobytes() == b'123456789\x00'