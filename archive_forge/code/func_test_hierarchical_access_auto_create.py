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
def test_hierarchical_access_auto_create(tmp_local_or_remote_netcdf):
    ds = h5netcdf.File(tmp_local_or_remote_netcdf, 'w')
    ds.create_variable('/foo/bar', data=1)
    g = ds.create_group('foo/baz')
    g.create_variable('/foo/hello', data=2)
    assert set(ds) == set(['foo'])
    assert set(ds['foo']) == set(['bar', 'baz', 'hello'])
    ds.close()
    ds = h5netcdf.File(tmp_local_or_remote_netcdf, 'r')
    assert set(ds) == set(['foo'])
    assert set(ds['foo']) == set(['bar', 'baz', 'hello'])
    ds.close()