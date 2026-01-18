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
def check_invalid_netcdf4(var, i):
    pdim = 'phony_dim_{}'
    assert var['foo1'].dimensions[0] == pdim.format(i * 4)
    assert var['foo1'].dimensions[1] == pdim.format(1 + i * 4)
    assert var['foo1'].dimensions[2] == pdim.format(2 + i * 4)
    assert var['foo2'].dimensions[0] == pdim.format(3 + i * 4)
    assert var['foo2'].dimensions[1] == pdim.format(0 + i * 4)
    assert var['foo2'].dimensions[2] == pdim.format(1 + i * 4)
    assert var['foo3'].dimensions[0] == pdim.format(i * 4)
    assert var['foo3'].dimensions[1] == pdim.format(1 + i * 4)
    assert var['foo3'].dimensions[2] == pdim.format(2 + i * 4)
    assert var['foo4'].dimensions[0] == pdim.format(3 + i * 4)
    assert var['foo4'].dimensions[1] == pdim.format(i * 4)
    assert var['foo4'].dimensions[2] == pdim.format(1 + i * 4)
    assert var['x'].dimensions[0] == pdim.format(i * 4)
    assert var['y'].dimensions[0] == pdim.format(i * 4)
    assert var['z'].dimensions[0] == pdim.format(i * 4)
    assert var['x1'].dimensions[0] == pdim.format(3 + i * 4)
    assert var['y1'].dimensions[0] == pdim.format(i * 4)
    assert var['z1'].dimensions[0] == pdim.format(i * 4)