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
def test_a2f_bad_scaling():
    NUMERICAL_TYPES = sum([sctypes[key] for key in ['int', 'uint', 'float', 'complex']], [])
    for in_type, out_type, slope, inter in itertools.product(NUMERICAL_TYPES, NUMERICAL_TYPES, (None, 1, 0, np.nan, -np.inf, np.inf), (0, np.nan, -np.inf, np.inf)):
        arr = np.ones((2,), dtype=in_type)
        fobj = BytesIO()
        cm = error_warnings()
        if np.issubdtype(in_type, np.complexfloating) and (not np.issubdtype(out_type, np.complexfloating)):
            cm = pytest.warns(ComplexWarning)
        if (slope, inter) == (1, 0):
            with cm:
                assert_array_equal(arr, write_return(arr, fobj, out_type, intercept=inter, divslope=slope))
        elif (slope, inter) == (None, 0):
            assert_array_equal(0, write_return(arr, fobj, out_type, intercept=inter, divslope=slope))
        else:
            with pytest.raises(ValueError):
                array_to_file(arr, fobj, np.int8, intercept=inter, divslope=slope)