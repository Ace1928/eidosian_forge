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
def test_compression():
    arr = np.zeros(100).reshape((5, 20))
    arr[2, 10] = 1
    stream = BytesIO()
    savemat(stream, {'arr': arr})
    raw_len = len(stream.getvalue())
    vals = loadmat(stream)
    assert_array_equal(vals['arr'], arr)
    stream = BytesIO()
    savemat(stream, {'arr': arr}, do_compression=True)
    compressed_len = len(stream.getvalue())
    vals = loadmat(stream)
    assert_array_equal(vals['arr'], arr)
    assert_(raw_len > compressed_len)
    arr2 = arr.copy()
    arr2[0, 0] = 1
    stream = BytesIO()
    savemat(stream, {'arr': arr, 'arr2': arr2}, do_compression=False)
    vals = loadmat(stream)
    assert_array_equal(vals['arr2'], arr2)
    stream = BytesIO()
    savemat(stream, {'arr': arr, 'arr2': arr2}, do_compression=True)
    vals = loadmat(stream)
    assert_array_equal(vals['arr2'], arr2)