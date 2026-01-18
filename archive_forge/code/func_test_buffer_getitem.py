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
def test_buffer_getitem():
    data = bytearray(b'some data!')
    buf = pa.py_buffer(data)
    n = len(data)
    for ix in range(-n, n - 1):
        assert buf[ix] == data[ix]
    with pytest.raises(IndexError):
        buf[n]
    with pytest.raises(IndexError):
        buf[-n - 1]