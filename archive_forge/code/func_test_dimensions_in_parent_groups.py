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
def test_dimensions_in_parent_groups(tmpdir):
    with netCDF4.Dataset(tmpdir.join('test_netcdf.nc'), mode='w') as ds:
        ds0 = ds
        for i in range(10):
            ds = ds.createGroup(f'group{i:02d}')
        ds0.createDimension('x', 10)
        ds0.createDimension('y', 20)
        ds0['group00'].createVariable('test', float, ('x', 'y'))
        var = ds0['group00'].createVariable('x', float, ('x', 'y'))
        var[:] = np.ones((10, 20))
    with legacyapi.Dataset(tmpdir.join('test_legacy.nc'), mode='w') as ds:
        ds0 = ds
        for i in range(10):
            ds = ds.createGroup(f'group{i:02d}')
        ds0.createDimension('x', 10)
        ds0.createDimension('y', 20)
        ds0['group00'].createVariable('test', float, ('x', 'y'))
        var = ds0['group00'].createVariable('x', float, ('x', 'y'))
        var[:] = np.ones((10, 20))
    with h5netcdf.File(tmpdir.join('test_netcdf.nc'), mode='r') as ds0:
        with h5netcdf.File(tmpdir.join('test_legacy.nc'), mode='r') as ds1:
            assert repr(ds0.dimensions['x']) == repr(ds1.dimensions['x'])
            assert repr(ds0.dimensions['y']) == repr(ds1.dimensions['y'])
            assert repr(ds0['group00']) == repr(ds1['group00'])
            assert repr(ds0['group00']['test']) == repr(ds1['group00']['test'])
            assert repr(ds0['group00']['x']) == repr(ds1['group00']['x'])