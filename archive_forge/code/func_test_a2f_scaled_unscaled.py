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
def test_a2f_scaled_unscaled():
    fobj = BytesIO()
    for in_dtype, out_dtype, intercept, divslope in itertools.product(NUMERIC_TYPES, NUMERIC_TYPES, (0, 0.5, -1, 1), (1, 0.5, 2)):
        mn_in, mx_in = _dt_min_max(in_dtype)
        vals = [mn_in, 0, 1, mx_in]
        if np.dtype(in_dtype).kind != 'u':
            vals.append(-1)
        if in_dtype in CFLOAT_TYPES:
            vals.append(np.nan)
        arr = np.array(vals, dtype=in_dtype)
        mn_out, mx_out = _dt_min_max(out_dtype)
        nan_fill = -intercept / divslope
        if out_dtype in IUINT_TYPES:
            nan_fill = np.round(nan_fill)
        if in_dtype in CFLOAT_TYPES and (not mn_out <= nan_fill <= mx_out):
            with pytest.raises(ValueError):
                array_to_file(arr, fobj, out_dtype=out_dtype, divslope=divslope, intercept=intercept)
            continue
        with suppress_warnings():
            back_arr = write_return(arr, fobj, out_dtype=out_dtype, divslope=divslope, intercept=intercept)
            exp_back = arr.copy()
            if in_dtype in IUINT_TYPES and out_dtype in IUINT_TYPES and ((intercept, divslope) == (0, 1)):
                if (mn_in, mx_in) != (mn_out, mx_out):
                    exp_back = np.clip(exp_back, max(mn_in, mn_out), min(mx_in, mx_out))
            else:
                if in_dtype in CFLOAT_TYPES and out_dtype in IUINT_TYPES:
                    exp_back[np.isnan(exp_back)] = 0
                if in_dtype not in COMPLEX_TYPES:
                    exp_back = exp_back.astype(float)
                if intercept != 0:
                    exp_back -= intercept
                if divslope != 1:
                    exp_back /= divslope
                if exp_back.dtype.type in CFLOAT_TYPES and out_dtype in IUINT_TYPES:
                    exp_back = np.round(exp_back).astype(float)
                    exp_back = np.clip(exp_back, *shared_range(float, out_dtype))
            exp_back = exp_back.astype(out_dtype)
        assert_allclose_safely(back_arr, exp_back)