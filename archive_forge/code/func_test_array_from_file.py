import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_array_from_file():
    shape = (2, 3, 4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    offset = 0
    assert buf_chk(in_arr, BytesIO(), None, offset)
    offset = 10
    assert buf_chk(in_arr, BytesIO(), None, offset)
    fname = 'test.bin'
    with InTemporaryDirectory():
        out_buf = open(fname, 'wb')
        in_buf = open(fname, 'rb')
        assert buf_chk(in_arr, out_buf, in_buf, offset)
        out_buf.seek(0)
        in_buf.seek(0)
        offset = 5
        assert buf_chk(in_arr, out_buf, in_buf, offset)
        del out_buf, in_buf
    arr = array_from_file((), np.dtype('f8'), BytesIO())
    assert len(arr) == 0
    arr = array_from_file((0,), np.dtype('f8'), BytesIO())
    assert len(arr) == 0
    with pytest.raises(OSError):
        array_from_file(shape, dtype, BytesIO())
    fd, fname = tempfile.mkstemp()
    with InTemporaryDirectory():
        open(fname, 'wb').write(b'1')
        in_buf = open(fname, 'rb')
        with pytest.raises(OSError):
            array_from_file(shape, dtype, in_buf)
        del in_buf