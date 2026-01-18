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
def test_invalid_netcdf_malformed_dimension_scales(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, 'w') as f:
        foo_data = np.arange(125).reshape(5, 5, 5)
        f.create_dataset('foo1', data=foo_data)
        f.create_dataset('x', data=np.arange(5))
        f.create_dataset('y', data=np.arange(5))
        f.create_dataset('z', data=np.arange(5))
        f['x'].make_scale()
        f['y'].make_scale()
        f['z'].make_scale()
        f['foo1'].dims[0].attach_scale(f['x'])
    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, 'r') as ds:
            assert ds
            print(ds)
    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='sort') as ds:
            assert ds
            print(ds)