import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
def test_nodata(self):
    nodata_filename = os.path.join(data_path, 'nodata.arff')
    data, meta = loadarff(nodata_filename)
    if sys.byteorder == 'big':
        end = '>'
    else:
        end = '<'
    expected_dtype = np.dtype([('sepallength', f'{end}f8'), ('sepalwidth', f'{end}f8'), ('petallength', f'{end}f8'), ('petalwidth', f'{end}f8'), ('class', 'S15')])
    assert_equal(data.dtype, expected_dtype)
    assert_equal(data.size, 0)