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
def test_bytes_reader():
    data = b'some sample data'
    f = pa.BufferReader(data)
    assert f.tell() == 0
    assert f.size() == len(data)
    assert f.read(4) == b'some'
    assert f.tell() == 4
    f.seek(0)
    assert f.tell() == 0
    f.seek(0, 2)
    assert f.tell() == len(data)
    f.seek(5)
    assert f.tell() == 5
    assert f.read(50) == b'sample data'
    assert not f.closed
    f.close()
    assert f.closed