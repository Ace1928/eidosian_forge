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
def check_netcdf_dimensions(tmp_netcdf, write_module, read_module):
    if read_module in [legacyapi, netCDF4]:
        opener = read_module.Dataset
    else:
        opener = h5netcdf.File
    with opener(tmp_netcdf, 'r') as ds:
        for i, grp in enumerate(['dimtest0', 'dimtest1']):
            g = ds.groups[grp]
            assert set(g.dimensions) == {'collide', 'ship_strlen', 'time', 'nvec', 'ship', 'sample'}
            if read_module in [legacyapi, h5netcdf]:
                assert g.dimensions['time'].isunlimited()
                assert g.dimensions['time'].size == 10 + i
                assert not g.dimensions['nvec'].isunlimited()
                assert g.dimensions['nvec'].size == 5 + i
                assert not g.dimensions['sample'].isunlimited()
                assert g.dimensions['sample'].size == 2 + i
                assert not g.dimensions['collide'].isunlimited()
                assert g.dimensions['collide'].size == 7 + i
                assert not g.dimensions['ship'].isunlimited()
                assert g.dimensions['ship'].size == 3 + i
                assert not g.dimensions['ship_strlen'].isunlimited()
                assert g.dimensions['ship_strlen'].size == 10
            else:
                assert g.dimensions['time'].isunlimited()
                assert g.dimensions['time'].size == 10 + i
                assert not g.dimensions['nvec'].isunlimited()
                assert g.dimensions['nvec'].size == 5 + i
                assert not g.dimensions['sample'].isunlimited()
                assert g.dimensions['sample'].size == 2 + i
                assert not g.dimensions['ship'].isunlimited()
                assert g.dimensions['ship'].size == 3 + i
                assert not g.dimensions['ship_strlen'].isunlimited()
                assert g.dimensions['ship_strlen'].size == 10
                assert not g.dimensions['collide'].isunlimited()
                assert g.dimensions['collide'].size == 7 + i
            assert set(g.variables) == {'data', 'collide', 'non_collide', 'time', 'sample', 'ship'}
            assert g.variables['time'].shape == (10 + i,)
            assert g.variables['data'].shape == (3 + i, 2 + i, 10 + i, 5 + i)
            assert g.variables['collide'].shape == (5 + i,)
            assert g.variables['non_collide'].shape == (5 + i,)
            assert g.variables['sample'].shape == (10 + i, 2 + i)
            assert g.variables['ship'].shape == (3 + i, 10)