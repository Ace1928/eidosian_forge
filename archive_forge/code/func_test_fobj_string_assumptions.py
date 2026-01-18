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
def test_fobj_string_assumptions():
    dtype = np.dtype(np.int32)

    def make_array(n, bytes):
        arr = np.ndarray(n, dtype, buffer=bytes)
        arr.flags.writeable = True
        return arr
    fname = 'test.bin'
    with InTemporaryDirectory():
        openers = [open, gzip.open, BZ2File]
        if HAVE_ZSTD:
            openers += [pyzstd.ZstdFile]
        for n, opener in itertools.product((256, 1024, 2560, 25600), openers):
            in_arr = np.arange(n, dtype=dtype)
            fobj_w = opener(fname, 'wb')
            fobj_w.write(in_arr.tobytes())
            fobj_w.close()
            fobj_r = opener(fname, 'rb')
            try:
                contents1 = bytearray(4 * n)
                fobj_r.readinto(contents1)
                assert contents1[0:8] != b'\x00' * 8
                out_arr = make_array(n, contents1)
                assert_array_equal(in_arr, out_arr)
                out_arr[1] = 0
                assert contents1[:8] == b'\x00' * 8
                fobj_r.seek(0)
                contents2 = bytearray(4 * n)
                fobj_r.readinto(contents2)
                out_arr2 = make_array(n, contents2)
                assert_array_equal(in_arr, out_arr2)
                assert out_arr[1] == 0
            finally:
                fobj_r.close()
            os.unlink(fname)