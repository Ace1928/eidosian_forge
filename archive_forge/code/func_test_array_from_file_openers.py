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
def test_array_from_file_openers():
    shape = (2, 3, 4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    with InTemporaryDirectory():
        extensions = ['', '.gz', '.bz2']
        if HAVE_ZSTD:
            extensions += ['.zst']
        for ext, offset in itertools.product(extensions, (0, 5, 10)):
            fname = 'test.bin' + ext
            with Opener(fname, 'wb') as out_buf:
                if offset != 0:
                    out_buf.write(b' ' * offset)
                out_buf.write(in_arr.tobytes(order='F'))
            with Opener(fname, 'rb') as in_buf:
                out_arr = array_from_file(shape, dtype, in_buf, offset)
                assert_array_almost_equal(in_arr, out_arr)
            del out_arr