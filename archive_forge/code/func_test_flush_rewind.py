import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
def test_flush_rewind():
    stream = BytesIO()
    with make_simple(stream, mode='w') as f:
        f.createDimension('x', 4)
        v = f.createVariable('v', 'i2', ['x'])
        v[:] = 1
        f.flush()
        len_single = len(stream.getvalue())
        f.flush()
        len_double = len(stream.getvalue())
    assert_(len_single == len_double)