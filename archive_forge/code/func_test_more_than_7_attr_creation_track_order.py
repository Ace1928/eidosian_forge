import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
@pytest.mark.parametrize('track_order', [False, True])
def test_more_than_7_attr_creation_track_order(tmp_local_netcdf, track_order):
    h5py_version = version.parse(h5py.__version__)
    if track_order and h5py_version < version.parse('3.7.0'):
        expected_errors = pytest.raises(KeyError)
    else:
        expected_errors = memoryview(b'')
    with h5netcdf.File(tmp_local_netcdf, 'w', track_order=track_order) as h5file:
        with expected_errors:
            for i in range(100):
                h5file.attrs[f'key{i}'] = i
                h5file.attrs[f'key{i}'] = 0