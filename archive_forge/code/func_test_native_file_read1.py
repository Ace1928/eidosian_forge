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
def test_native_file_read1(tmpdir):
    data = b'123\n' * 1000000
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data)
    chunks = []
    with pa.OSFile(path, mode='rb') as f:
        while True:
            b = f.read1()
            assert len(b) < len(data)
            chunks.append(b)
            b = f.read1(30000)
            assert len(b) <= 30000
            chunks.append(b)
            if not b:
                break
    assert b''.join(chunks) == data