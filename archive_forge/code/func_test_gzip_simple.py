import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
def test_gzip_simple():
    xdense = np.zeros((20, 20))
    xdense[2, 3] = 2.3
    xdense[4, 5] = 4.5
    x = SP.csc_matrix(xdense)
    name = 'gzip_test'
    expected = {'x': x}
    format = '4'
    tmpdir = mkdtemp()
    try:
        fname = pjoin(tmpdir, name)
        mat_stream = gzip.open(fname, mode='wb')
        savemat(mat_stream, expected, format=format)
        mat_stream.close()
        mat_stream = gzip.open(fname, mode='rb')
        actual = loadmat(mat_stream, struct_as_record=True)
        mat_stream.close()
    finally:
        shutil.rmtree(tmpdir)
    assert_array_almost_equal(actual['x'].toarray(), expected['x'].toarray(), err_msg=repr(actual))