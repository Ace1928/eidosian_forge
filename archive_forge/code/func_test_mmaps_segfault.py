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
def test_mmaps_segfault():
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    if not IS_PYPY:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with netcdf_file(filename, mmap=True) as f:
                x = f.variables['lat'][:]
                del x

    def doit():
        with netcdf_file(filename, mmap=True) as f:
            return f.variables['lat'][:]
    with suppress_warnings() as sup:
        message = 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist'
        sup.filter(RuntimeWarning, message)
        x = doit()
    x.sum()