import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
@contextmanager
def make_simple(*args, **kwargs):
    f = netcdf_file(*args, **kwargs)
    f.history = 'Created for a test'
    f.createDimension('time', N_EG_ELS)
    time = f.createVariable('time', VARTYPE_EG, ('time',))
    time[:] = np.arange(N_EG_ELS)
    time.units = 'days since 2008-01-01'
    f.flush()
    yield f
    f.close()