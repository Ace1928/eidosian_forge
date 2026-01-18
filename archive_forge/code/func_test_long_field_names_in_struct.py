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
def test_long_field_names_in_struct():
    lim = 63
    fldname = 'a' * lim
    cell = np.ndarray((1, 2), dtype=object)
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    cell[0, 0] = st1
    cell[0, 1] = st1
    savemat(BytesIO(), {'longstruct': cell}, format='5', long_field_names=True)
    assert_raises(ValueError, savemat, BytesIO(), {'longstruct': cell}, format='5', long_field_names=False)