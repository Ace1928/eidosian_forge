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
def test_buffer_address():
    b1 = b'some data!'
    b2 = bytearray(b1)
    b3 = bytearray(b1)
    buf1 = pa.py_buffer(b1)
    buf2 = pa.py_buffer(b1)
    buf3 = pa.py_buffer(b2)
    buf4 = pa.py_buffer(b3)
    assert buf1.address > 0
    assert buf1.address == buf2.address
    assert buf3.address != buf2.address
    assert buf4.address != buf3.address
    arr = np.arange(5)
    buf = pa.py_buffer(arr)
    assert buf.address == arr.ctypes.data