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
def test_encoded_fill_value():
    with netcdf_file(BytesIO(), mode='w') as f:
        f.createDimension('x', 1)
        var = f.createVariable('var', 'S1', ('x',))
        assert_equal(var._get_encoded_fill_value(), b'\x00')
        var._FillValue = b'\x01'
        assert_equal(var._get_encoded_fill_value(), b'\x01')
        var._FillValue = b'\x00\x00'
        assert_equal(var._get_encoded_fill_value(), b'\x00')