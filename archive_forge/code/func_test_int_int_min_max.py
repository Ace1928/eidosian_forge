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
def test_int_int_min_max():
    eps = np.finfo(np.float64).eps
    rtol = 1e-06
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        arr = np.array([iinf.min, iinf.max], dtype=in_dt)
        for out_dt in IUINT_TYPES:
            try:
                aw = SlopeInterArrayWriter(arr, out_dt)
            except ScalingError:
                continue
            arr_back_sc = round_trip(aw)
            adiff = int_abs(arr - arr_back_sc)
            rdiff = adiff / (arr + eps)
            assert np.all(rdiff < rtol)