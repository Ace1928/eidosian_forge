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
def check_compressed_input(data, fn, compression):
    raw = pa.OSFile(fn, mode='rb')
    with pa.CompressedInputStream(raw, compression) as compressed:
        assert not compressed.closed
        assert compressed.readable()
        assert not compressed.writable()
        assert not compressed.seekable()
        got = compressed.read()
        assert got == data
    assert compressed.closed
    assert raw.closed
    raw = pa.OSFile(fn, mode='rb')
    with pa.CompressedInputStream(raw, compression) as compressed:
        buf = compressed.read_buffer()
        assert isinstance(buf, pa.Buffer)
        assert buf.to_pybytes() == data