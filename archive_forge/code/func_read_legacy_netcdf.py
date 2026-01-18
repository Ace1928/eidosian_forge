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
def read_legacy_netcdf(tmp_netcdf, read_module, write_module):
    ds = read_module.Dataset(tmp_netcdf, 'r')
    assert ds.ncattrs() == ['global', 'other_attr']
    assert ds.getncattr('global') == 42
    if write_module is not netCDF4:
        assert ds.other_attr == 'yes'
    with pytest.raises(AttributeError):
        ds.does_not_exist
    assert set(ds.dimensions) == set(['x', 'y', 'z', 'empty', 'string3', 'mismatched_dim', 'unlimited'])
    assert set(ds.variables) == set(['foo', 'y', 'z', 'intscalar', 'scalar', 'var_len_str', 'mismatched_dim', 'foo_unlimited'])
    assert set(ds.groups) == set(['subgroup'])
    assert ds.parent is None
    v = ds.variables['foo']
    assert array_equal(v, np.ones((4, 5)))
    assert v.dtype == float
    assert v.dimensions == ('x', 'y')
    assert v.ndim == 2
    assert v.ncattrs() == ['units']
    if write_module is not netCDF4:
        assert v.getncattr('units') == 'meters'
    assert tuple(v.chunking()) == (4, 5)
    filters = v.filters()
    assert filters['complevel'] == 4
    assert filters['fletcher32'] is False
    assert filters['shuffle'] is True
    assert filters['zlib'] is True
    v = ds.variables['y']
    assert array_equal(v, np.r_[np.arange(4), [-1]])
    assert v.dtype == int
    assert v.dimensions == ('y',)
    assert v.ndim == 1
    assert v.ncattrs() == ['_FillValue']
    assert v.getncattr('_FillValue') == -1
    assert v.chunking() == 'contiguous'
    filters = v.filters()
    assert filters['complevel'] == 0
    assert filters['fletcher32'] is False
    assert filters['shuffle'] is False
    assert filters['zlib'] is False
    ds.close()
    if is_h5py_char_working(tmp_netcdf, 'z'):
        ds = read_module.Dataset(tmp_netcdf, 'r')
        v = ds.variables['z']
        assert array_equal(v, _char_array)
        assert v.dtype == 'S1'
        assert v.ndim == 2
        assert v.dimensions == ('z', 'string3')
        assert v.ncattrs() == ['_FillValue']
        assert v.getncattr('_FillValue') == b'X'
    else:
        ds = read_module.Dataset(tmp_netcdf, 'r')
    v = ds.variables['scalar']
    assert array_equal(v, np.array(2.0))
    assert v.dtype == 'float32'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []
    v = ds.variables['intscalar']
    assert array_equal(v, np.array(2))
    assert v.dtype == 'int64'
    assert v.ndim == 0
    assert v.dimensions == ()
    assert v.ncattrs() == []
    v = ds.variables['var_len_str']
    assert v.dtype == str
    assert v[0] == _vlen_string
    v = ds.groups['subgroup'].variables['subvar']
    assert ds.groups['subgroup'].parent is ds
    assert array_equal(v, np.arange(4.0))
    assert v.dtype == 'int32'
    assert v.ndim == 1
    assert v.dimensions == ('x',)
    assert v.ncattrs() == []
    v = ds.groups['subgroup'].variables['y_var']
    assert v.shape == (10,)
    assert 'y' in ds.groups['subgroup'].dimensions
    ds.close()