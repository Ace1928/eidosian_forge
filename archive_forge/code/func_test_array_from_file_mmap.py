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
def test_array_from_file_mmap():
    shape = (2, 21)
    with InTemporaryDirectory():
        for dt in (np.int16, np.float64):
            arr = np.arange(np.prod(shape), dtype=dt).reshape(shape)
            with open('test.bin', 'wb') as fobj:
                fobj.write(arr.tobytes(order='F'))
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj)
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap=True)
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='c')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'c'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='r')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'r'
            with open('test.bin', 'rb+') as fobj:
                res = array_from_file(shape, dt, fobj, mmap='r+')
                assert_array_equal(res, arr)
                assert isinstance(res, np.memmap)
                assert res.mode == 'r+'
            with open('test.bin', 'rb') as fobj:
                res = array_from_file(shape, dt, fobj, mmap=False)
                assert_array_equal(res, arr)
                assert not isinstance(res, np.memmap)
            with open('test.bin', 'rb') as fobj:
                with pytest.raises(ValueError):
                    array_from_file(shape, dt, fobj, mmap='p')