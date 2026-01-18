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
def test_empty_sparse():
    sio = BytesIO()
    import scipy.sparse
    empty_sparse = scipy.sparse.csr_matrix([[0, 0], [0, 0]])
    savemat(sio, dict(x=empty_sparse))
    sio.seek(0)
    res = loadmat(sio)
    assert_array_equal(res['x'].shape, empty_sparse.shape)
    assert_array_equal(res['x'].toarray(), 0)
    sio.seek(0)
    reader = MatFile5Reader(sio)
    reader.initialize_read()
    reader.read_file_header()
    hdr, _ = reader.read_var_header()
    assert_equal(hdr.nzmax, 1)