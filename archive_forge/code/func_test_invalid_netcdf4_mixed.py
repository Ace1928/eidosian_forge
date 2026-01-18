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
def test_invalid_netcdf4_mixed(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip('netCDF4 package does not work with remote HDF5 files')
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, 'w') as f:
        var, var2 = create_invalid_netcdf_data()
        for k, v in var.items():
            f.create_dataset(k, data=v)
        for k, v in var2.items():
            f.create_dataset(k, data=np.arange(v))
        f['x1'].make_scale()
        f['y1'].make_scale()
        f['z1'].make_scale()
        f['foo2'].dims[0].attach_scale(f['x1'])
        f['foo2'].dims[1].attach_scale(f['y1'])
        f['foo2'].dims[2].attach_scale(f['z1'])
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='sort') as ds:
        var = ds.variables
        check_invalid_netcdf4_mixed(var, 3)
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='access') as ds:
        var = ds.variables
        check_invalid_netcdf4_mixed(var, 0)
    if not tmp_local_or_remote_netcdf.startswith(remote_h5):
        with netCDF4.Dataset(tmp_local_or_remote_netcdf, 'r') as ds:
            var = ds.variables
            check_invalid_netcdf4_mixed(var, 3)
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r') as ds:
        with raises(ValueError):
            ds.variables['foo1'].dimensions