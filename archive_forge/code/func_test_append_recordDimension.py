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
def test_append_recordDimension():
    dataSize = 100
    with in_tempdir():
        with netcdf_file('withRecordDimension.nc', 'w') as f:
            f.createDimension('time', None)
            f.createVariable('time', 'd', ('time',))
            f.createDimension('x', dataSize)
            x = f.createVariable('x', 'd', ('x',))
            x[:] = np.array(range(dataSize))
            f.createDimension('y', dataSize)
            y = f.createVariable('y', 'd', ('y',))
            y[:] = np.array(range(dataSize))
            f.createVariable('testData', 'i', ('time', 'x', 'y'))
            f.flush()
            f.close()
        for i in range(2):
            with netcdf_file('withRecordDimension.nc', 'a') as f:
                f.variables['time'].data = np.append(f.variables['time'].data, i)
                f.variables['testData'][i, :, :] = np.full((dataSize, dataSize), i)
                f.flush()
            with netcdf_file('withRecordDimension.nc') as f:
                assert_equal(f.variables['time'][-1], i)
                assert_equal(f.variables['testData'][-1, :, :].copy(), np.full((dataSize, dataSize), i))
                assert_equal(f.variables['time'].data.shape[0], i + 1)
                assert_equal(f.variables['testData'].data.shape[0], i + 1)
        with netcdf_file('withRecordDimension.nc') as f:
            with assert_raises(KeyError) as ar:
                f.variables['testData']._attributes['data']
            ex = ar.value
            assert_equal(ex.args[0], 'data')