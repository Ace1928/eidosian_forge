import os
from os.path import join as pjoin, dirname
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
import pytest
from ..netcdf import netcdf_file
def assert_simple_truths(ncfileobj):
    assert ncfileobj.history == b'Created for a test'
    time = ncfileobj.variables['time']
    assert time.units == b'days since 2008-01-01'
    assert time.shape == (N_EG_ELS,)
    assert time[-1] == N_EG_ELS - 1