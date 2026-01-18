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
def test_buffer_eq_bytes():
    buf = pa.py_buffer(b'some data')
    assert buf == b'some data'
    assert buf == bytearray(b'some data')
    assert buf != b'some dat1'
    with pytest.raises(TypeError):
        buf == 'some data'