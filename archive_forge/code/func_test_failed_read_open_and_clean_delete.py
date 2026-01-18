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
def test_failed_read_open_and_clean_delete(tmpdir):
    path = str(tmpdir.join('this_file_does_not_exist.nc'))
    try:
        with h5netcdf.File(path, 'r') as ds:
            assert ds
    except OSError:
        pass
    obj_list = gc.get_objects()
    for obj in obj_list:
        try:
            is_h5netcdf_File = isinstance(obj, h5netcdf.File)
        except AttributeError:
            is_h5netcdf_File = False
        if is_h5netcdf_File:
            obj.close()