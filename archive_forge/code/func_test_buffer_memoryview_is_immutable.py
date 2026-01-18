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
def test_buffer_memoryview_is_immutable():
    val = b'some data'
    buf = pa.py_buffer(val)
    assert not buf.is_mutable
    assert isinstance(buf, pa.Buffer)
    result = memoryview(buf)
    assert result.readonly
    with pytest.raises(TypeError) as exc:
        result[0] = b'h'
        assert 'cannot modify read-only' in str(exc.value)
    b = bytes(buf)
    with pytest.raises(TypeError) as exc:
        b[0] = b'h'
        assert 'cannot modify read-only' in str(exc.value)