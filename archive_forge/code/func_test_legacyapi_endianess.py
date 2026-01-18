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
def test_legacyapi_endianess(tmp_local_netcdf):
    big = legacyapi._check_return_dtype_endianess('big')
    little = legacyapi._check_return_dtype_endianess('little')
    native = legacyapi._check_return_dtype_endianess('native')
    with legacyapi.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('x', 4)
        v = ds.createVariable('big', int, 'x', endian='big')
        v[...] = 65533
        v = ds.createVariable('little', int, 'x', endian='little')
        v[...] = 65533
        v = ds.createVariable('native', int, 'x', endian='native')
        v[...] = 65535
    with h5py.File(tmp_local_netcdf, 'r') as ds:
        assert ds['big'].dtype.byteorder == big
        assert ds['little'].dtype.byteorder == little
        assert ds['native'].dtype.byteorder == native
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds['big'].dtype.byteorder == big
        assert ds['little'].dtype.byteorder == little
        assert ds['native'].dtype.byteorder == native
    with legacyapi.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['big'].dtype.byteorder == big
        assert ds['little'].dtype.byteorder == little
        assert ds['native'].dtype.byteorder == native
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['big'].dtype.byteorder == big
        assert ds['little'].dtype.byteorder == little
        assert ds['native'].dtype.byteorder == native