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
def create_invalid_netcdf_data():
    foo_data = np.arange(125).reshape(5, 5, 5)
    bar_data = np.arange(625).reshape(25, 5, 5)
    var = {'foo1': foo_data, 'foo2': bar_data, 'foo3': foo_data, 'foo4': bar_data}
    var2 = {'x': 5, 'y': 5, 'z': 5, 'x1': 25, 'y1': 5, 'z1': 5}
    return (var, var2)