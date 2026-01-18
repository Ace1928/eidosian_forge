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
def test_get_dim_scale_refs(tmp_local_netcdf):
    with legacyapi.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('x', 10)
        ds.createVariable('test0', 'i8', ('x',))
        ds.createVariable('test1', 'i8', ('x',))
    with legacyapi.Dataset(tmp_local_netcdf, 'r') as ds:
        refs = ds.dimensions['x']._scale_refs
        assert ds._h5file[refs[0][0]] == ds['test0']._h5ds
        assert ds._h5file[refs[1][0]] == ds['test1']._h5ds