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
def test_nativefile_write_memoryview():
    f = pa.BufferOutputStream()
    data = b'ok'
    arr = np.frombuffer(data, dtype='S1')
    f.write(arr)
    f.write(bytearray(data))
    f.write(pa.py_buffer(data))
    with pytest.raises(TypeError):
        f.write(data.decode('utf8'))
    buf = f.getvalue()
    assert buf.to_pybytes() == data * 3