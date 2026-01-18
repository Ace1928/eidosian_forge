import itertools
from io import BytesIO
from platform import machine, python_compiler
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..arraywriters import (
from ..casting import int_abs, sctypes, shared_range, type_info
from ..testing import assert_allclose_safely, suppress_warnings
from ..volumeutils import _dt_min_max, apply_read_scaling, array_from_file
def test_io_scaling():
    bio = BytesIO()
    for in_type, out_type in itertools.product((np.int16, np.uint16, np.float32), (np.int8, np.uint8, np.int16, np.uint16)):
        out_dtype = np.dtype(out_type)
        info = type_info(in_type)
        imin, imax = (info['min'], info['max'])
        if imin == 0:
            val_tuples = ((0, imax), (100, imax))
        else:
            val_tuples = ((imin, 0, imax), (imin, 0), (0, imax), (imin, 100, imax))
        if imin != 0:
            val_tuples += ((imin, 0), (0, imax))
        for vals in val_tuples:
            arr = np.array(vals, dtype=in_type)
            aw = SlopeInterArrayWriter(arr, out_dtype)
            aw.to_fileobj(bio)
            arr2 = array_from_file(arr.shape, out_dtype, bio)
            arr3 = apply_read_scaling(arr2, aw.slope, aw.inter)
            max_miss = np.abs(aw.slope) / 2.0
            abs_err = np.abs(arr - arr3)
            assert np.all(abs_err <= max_miss)
            if out_type in UINT_TYPES and 0 in (min(arr), max(arr)):
                assert min(abs_err) == abs_err[arr == 0]
            bio.truncate(0)
            bio.seek(0)