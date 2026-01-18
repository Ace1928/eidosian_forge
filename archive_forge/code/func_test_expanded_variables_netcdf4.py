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
def test_expanded_variables_netcdf4(tmp_local_netcdf, netcdf_write_module):
    with netcdf_write_module.Dataset(tmp_local_netcdf, 'w') as ds:
        f = ds.createGroup('test')
        f.createDimension('x', None)
        f.createDimension('y', 3)
        dummy1 = f.createVariable('dummy1', float, ('x', 'y'))
        dummy2 = f.createVariable('dummy2', float, ('x', 'y'))
        dummy3 = f.createVariable('dummy3', float, ('x', 'y'))
        dummy4 = f.createVariable('dummy4', float, ('x', 'y'))
        dummy1[:] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        dummy2[1, :] = [4, 5, 6]
        dummy3[0:2, :] = [[1, 2, 3], [4, 5, 6]]
        if netcdf_write_module == netCDF4:
            ds.set_auto_mask(False)
        res1 = dummy1[:]
        res2 = dummy2[:]
        res3 = dummy3[:]
        res4 = dummy4[:]
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as ds:
        if netcdf_write_module == netCDF4:
            ds.set_auto_mask(False)
        f = ds['test']
        np.testing.assert_allclose(f.variables['dummy1'][:], res1)
        np.testing.assert_allclose(f.variables['dummy1'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy1'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy1'].shape == (3, 3)
        np.testing.assert_allclose(f.variables['dummy2'][:], res2)
        np.testing.assert_allclose(f.variables['dummy2'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy2'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy2'].shape == (3, 3)
        np.testing.assert_allclose(f.variables['dummy3'][:], res3)
        np.testing.assert_allclose(f.variables['dummy3'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy3'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy3'].shape == (3, 3)
        np.testing.assert_allclose(f.variables['dummy4'][:], res4)
        assert f.variables['dummy4'].shape == (3, 3)
    with legacyapi.Dataset(tmp_local_netcdf, 'r') as ds:
        f = ds['test']
        np.testing.assert_allclose(f.variables['dummy1'][:], res1)
        np.testing.assert_allclose(f.variables['dummy1'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy1'][1:2, :], [[4.0, 5.0, 6.0]])
        np.testing.assert_allclose(f.variables['dummy1']._h5ds[1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy1']._h5ds[1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy1'].shape == (3, 3)
        assert f.variables['dummy1']._h5ds.shape == (3, 3)
        np.testing.assert_allclose(f.variables['dummy2'][:], res2)
        np.testing.assert_allclose(f.variables['dummy2'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy2'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy2'].shape == (3, 3)
        assert f.variables['dummy2']._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables['dummy3'][:], res3)
        np.testing.assert_allclose(f.variables['dummy3'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy3'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy3'].shape == (3, 3)
        assert f.variables['dummy3']._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables['dummy4'][:], res4)
        assert f.variables['dummy4'].shape == (3, 3)
        assert f.variables['dummy4']._h5ds.shape == (0, 3)
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        f = ds['test']
        np.testing.assert_allclose(f.variables['dummy1'][:], res1)
        np.testing.assert_allclose(f.variables['dummy1'][:, :], res1)
        np.testing.assert_allclose(f.variables['dummy1'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy1'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy1'].shape == (3, 3)
        assert f.variables['dummy1']._h5ds.shape == (3, 3)
        np.testing.assert_allclose(f.variables['dummy2'][:], res2)
        np.testing.assert_allclose(f.variables['dummy2'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy2'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy2'].shape == (3, 3)
        assert f.variables['dummy2']._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables['dummy3'][:], res3)
        np.testing.assert_allclose(f.variables['dummy3'][1, :], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(f.variables['dummy3'][1:2, :], [[4.0, 5.0, 6.0]])
        assert f.variables['dummy3'].shape == (3, 3)
        assert f.variables['dummy3']._h5ds.shape == (2, 3)
        np.testing.assert_allclose(f.variables['dummy4'][:], res4)
        assert f.variables['dummy4'].shape == (3, 3)
        assert f.variables['dummy4']._h5ds.shape == (0, 3)