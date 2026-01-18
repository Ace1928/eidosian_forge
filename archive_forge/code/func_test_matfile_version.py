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
@pytest.mark.parametrize('version, filt, regex', [(0, '_4*_*', None), (1, '_5*_*', None), (1, '_6*_*', None), (1, '_7*_*', '^((?!hdf5).)*$'), (2, '_7*_*', '.*hdf5.*'), (1, '8*_*', None)])
def test_matfile_version(version, filt, regex):
    use_filt = pjoin(test_data_path, 'test*%s.mat' % filt)
    files = glob(use_filt)
    if regex is not None:
        files = [file for file in files if re.match(regex, file) is not None]
    assert len(files) > 0, f'No files for version {version} using filter {filt}'
    for file in files:
        got_version = matfile_version(file)
        assert got_version[0] == version