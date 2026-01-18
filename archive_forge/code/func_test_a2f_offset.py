import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_a2f_offset():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    str_io = BytesIO()
    str_io.write(b'a' * 42)
    array_to_file(arr, str_io, np.float64, 42)
    data_back = array_from_file(arr.shape, np.float64, str_io, 42)
    assert_array_equal(data_back, arr.astype(np.float64))
    str_io.truncate(22)
    str_io.seek(22)
    array_to_file(arr, str_io, np.float64, None)
    data_back = array_from_file(arr.shape, np.float64, str_io, 22)
    assert_array_equal(data_back, arr.astype(np.float64))