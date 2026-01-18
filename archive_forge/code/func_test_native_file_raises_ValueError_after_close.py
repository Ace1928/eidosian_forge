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
def test_native_file_raises_ValueError_after_close(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(b'foooo')
    with pa.OSFile(path, mode='rb') as os_file:
        assert not os_file.closed
    assert os_file.closed
    with pa.memory_map(path, mode='rb') as mmap_file:
        assert not mmap_file.closed
    assert mmap_file.closed
    files = [os_file, mmap_file]
    methods = [('tell', ()), ('seek', (0,)), ('size', ()), ('flush', ()), ('readable', ()), ('writable', ()), ('seekable', ())]
    for f in files:
        for method, args in methods:
            with pytest.raises(ValueError):
                getattr(f, method)(*args)