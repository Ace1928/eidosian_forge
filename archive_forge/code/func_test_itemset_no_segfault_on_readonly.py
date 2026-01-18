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
def test_itemset_no_segfault_on_readonly():
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    with suppress_warnings() as sup:
        message = 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist'
        sup.filter(RuntimeWarning, message)
        with netcdf_file(filename, 'r', mmap=True) as f:
            time_var = f.variables['time']
    assert_raises(RuntimeError, time_var.assignValue, 42)