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
def test_buffer_weakref():
    buf = pa.py_buffer(b'some data')
    wr = weakref.ref(buf)
    assert wr() is not None
    del buf
    assert wr() is None