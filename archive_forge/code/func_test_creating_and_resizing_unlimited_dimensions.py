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
def test_creating_and_resizing_unlimited_dimensions(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w') as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 15
        f.dimensions['z'] = None
        f.resize_dimension('z', 20)
        with pytest.raises(ValueError) as e:
            f.resize_dimension('y', 20)
        assert e.value.args[0] == "Dimension 'y' is not unlimited and thus cannot be resized."
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, 'r') as f:
        assert f['x'].shape == (0,)
        assert f['x'].maxshape == (None,)
        assert f['y'].shape == (15,)
        assert f['y'].maxshape == (15,)
        assert f['z'].shape == (20,)
        assert f['z'].maxshape == (None,)