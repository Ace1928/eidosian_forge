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
def test_buffered_input_stream():
    raw = pa.BufferReader(b'123456789')
    f = pa.BufferedInputStream(raw, buffer_size=4)
    assert f.read(2) == b'12'
    assert raw.tell() == 4
    f.close()
    assert f.closed
    assert raw.closed