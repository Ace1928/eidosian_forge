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
def test_nc4_non_coord(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        f.dimensions = {'x': None, 'y': 2}
        f.create_variable('test', dimensions=('x',), dtype=np.int64)
        f.create_variable('y', dimensions=('x',), dtype=np.int64)
    with h5netcdf.File(tmp_local_netcdf, 'r') as f:
        assert list(f.dimensions) == ['x', 'y']
        assert f.dimensions['x'].size == 0
        assert f.dimensions['x'].isunlimited()
        assert f.dimensions['y'].size == 2
        if version.parse(h5py.__version__) >= version.parse('3.7.0'):
            assert list(f.variables) == ['test', 'y']
            assert list(f._h5group.keys()) == ['x', 'y', 'test', '_nc4_non_coord_y']
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        f.dimensions = {'x': None, 'y': 2}
        f.create_variable('y', dimensions=('x',), dtype=np.int64)
        f.create_variable('test', dimensions=('x',), dtype=np.int64)
    with h5netcdf.File(tmp_local_netcdf, 'r') as f:
        assert list(f.dimensions) == ['x', 'y']
        assert f.dimensions['x'].size == 0
        assert f.dimensions['x'].isunlimited()
        assert f.dimensions['y'].size == 2
        if version.parse(h5py.__version__) >= version.parse('3.7.0'):
            assert list(f.variables) == ['y', 'test']
            assert list(f._h5group.keys()) == ['x', 'y', '_nc4_non_coord_y', 'test']