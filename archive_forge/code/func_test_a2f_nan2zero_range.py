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
def test_a2f_nan2zero_range():
    fobj = BytesIO()
    for dt in INT_TYPES:
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mn=1, nan2zero=True)
        assert_array_equal([1, 1, 1, 2], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mx=-1, nan2zero=True)
        assert_array_equal([-1, -1, -1, -1], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8, intercept=129, nan2zero=True)
        assert_array_equal([-128, -128, -128, -127], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8, intercept=257.1, divslope=2, nan2zero=True)
        assert_array_equal([-128, -128, -128, -128], back_arr)
    for dt in CFLOAT_TYPES:
        arr = np.array([-1, 0, 1, np.nan], dtype=dt)
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        complex_warn = (ComplexWarning,) if np.issubdtype(dt, np.complexfloating) else ()
        nan_warn = (RuntimeWarning,) if FP_RUNTIME_WARN else ()
        c_and_n_warn = complex_warn + nan_warn
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            assert_array_equal([1, 1, 1, 0], write_return(arr, fobj, np.int8, mn=1))
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            assert_array_equal([-1, -1, -1, 0], write_return(arr, fobj, np.int8, mx=-1))
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=128)
        assert_array_equal([-128, -128, -127, -128], back_arr)
        with pytest.raises(ValueError):
            write_return(arr, fobj, np.int8, intercept=129)
        with pytest.raises(ValueError):
            write_return(arr_no_nan, fobj, np.int8, intercept=129)
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            nan_cast = np.array(np.nan, dtype=dt).astype(np.int8)
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=129, nan2zero=False)
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)
        with pytest.warns(complex_warn) if complex_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=256, divslope=2)
        assert_array_equal([-128, -128, -128, -128], back_arr)
        with pytest.raises(ValueError):
            write_return(arr, fobj, np.int8, intercept=257.1, divslope=2)
        with pytest.raises(ValueError):
            write_return(arr_no_nan, fobj, np.int8, intercept=257.1, divslope=2)
        with pytest.warns(c_and_n_warn) if c_and_n_warn else error_warnings():
            back_arr = write_return(arr, fobj, np.int8, intercept=257.1, divslope=2, nan2zero=False)
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)