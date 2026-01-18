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
def test_arraywriters():
    if machine() == 'sparc64' and python_compiler().startswith('GCC'):
        test_types = FLOAT_TYPES + IUINT_TYPES
    else:
        test_types = NUMERIC_TYPES
    for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
        for type in test_types:
            arr = np.arange(10, dtype=type)
            aw = klass(arr)
            assert aw.array is arr
            assert aw.out_dtype == arr.dtype
            assert_array_equal(arr, round_trip(aw))
            bs_arr = arr.byteswap()
            bs_arr = bs_arr.view(bs_arr.dtype.newbyteorder('S'))
            bs_aw = klass(bs_arr)
            bs_aw_rt = round_trip(bs_aw)
            assert_array_equal(arr, bs_aw_rt)
            bs_aw2 = klass(bs_arr, arr.dtype)
            bs_aw2_rt = round_trip(bs_aw2)
            assert_array_equal(arr, bs_aw2_rt)
            arr2 = np.reshape(arr, (2, 5))
            a2w = klass(arr2)
            arr_back = round_trip(a2w)
            assert_array_equal(arr2, arr_back)
            arr_back = round_trip(a2w, 'F')
            assert_array_equal(arr2, arr_back)
            arr_back = round_trip(a2w, 'C')
            assert_array_equal(arr2, arr_back)
            assert arr_back.flags.c_contiguous