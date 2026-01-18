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
def test_buffer_protocol_ref_counting():

    def make_buffer(bytes_obj):
        return bytearray(pa.py_buffer(bytes_obj))
    buf = make_buffer(b'foo')
    gc.collect()
    assert buf == b'foo'
    val = b'foo'
    refcount_before = sys.getrefcount(val)
    for i in range(10):
        make_buffer(val)
    gc.collect()
    assert refcount_before == sys.getrefcount(val)