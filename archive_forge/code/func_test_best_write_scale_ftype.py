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
def test_best_write_scale_ftype():
    for dtt in IUINT_TYPES + FLOAT_TYPES:
        arr = np.arange(10, dtype=dtt)
        assert best_write_scale_ftype(arr, 1, 0) == better_float_of(dtt, np.float32)
        assert best_write_scale_ftype(arr, 1, 0, np.float64) == better_float_of(dtt, np.float64)
        assert best_write_scale_ftype(arr, np.float32(2), 0) == better_float_of(dtt, np.float32)
        assert best_write_scale_ftype(arr, 1, np.float32(1)) == better_float_of(dtt, np.float32)
    best_vals = ((np.float32, np.float64),)
    if np.longdouble in OK_FLOATS:
        best_vals += ((np.float64, np.longdouble),)
    for lower_t, higher_t in best_vals:
        L_info = type_info(lower_t)
        t_max = L_info['max']
        nmant = L_info['nmant']
        big_delta = lower_t(2 ** (floor_log2(t_max) - nmant))
        arr = np.array([0, t_max], dtype=lower_t)
        assert best_write_scale_ftype(arr, 1, 0) == lower_t
        assert best_write_scale_ftype(arr, lower_t(1.01), 0) == lower_t
        assert best_write_scale_ftype(arr, lower_t(0.99), 0) == higher_t
        assert best_write_scale_ftype(arr, 1, -big_delta / 2.01) == lower_t
        assert best_write_scale_ftype(arr, 1, -big_delta / 2.0) == higher_t
        arr[0] = np.inf
        assert best_write_scale_ftype(arr, lower_t(0.5), 0) == lower_t
        arr[0] = -np.inf
        assert best_write_scale_ftype(arr, lower_t(0.5), 0) == lower_t