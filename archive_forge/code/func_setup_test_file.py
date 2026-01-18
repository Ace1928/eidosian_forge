import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
@contextmanager
def setup_test_file():
    val = b'a\x00string'
    fd, fname = mkstemp()
    with os.fdopen(fd, 'wb') as fs:
        fs.write(val)
    with open(fname, 'rb') as fs:
        gs = BytesIO(val)
        cs = BytesIO(val)
        yield (fs, gs, cs)
    os.unlink(fname)