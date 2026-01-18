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
def test_create_invalid_netcdf_catch_error(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        try:
            f.create_variable('test', ('x', 'y'), data=np.ones((10, 10), dtype='bool'))
        except CompatibilityError:
            pass
        assert repr(f.dimensions) == '<h5netcdf.Dimensions: >'