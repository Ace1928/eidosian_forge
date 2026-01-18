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
def test_memory_map_writer(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype('u1')
    data = arr.tobytes()[:SIZE]
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data)
    f = pa.memory_map(path, mode='r+b')
    f.seek(10)
    f.write(b'peekaboo')
    assert f.tell() == 18
    f.seek(10)
    assert f.read(8) == b'peekaboo'
    f2 = pa.memory_map(path, mode='r+b')
    f2.seek(10)
    f2.write(b'booapeak')
    f2.seek(10)
    f.seek(10)
    assert f.read(8) == b'booapeak'
    f3 = pa.memory_map(path, mode='w')
    f3.write(b'foo')
    with pa.memory_map(path) as f4:
        assert f4.size() == SIZE
    with pytest.raises(IOError):
        f3.read(5)
    f.seek(0)
    assert f.read(3) == b'foo'