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
def test_int_int_slope():
    eps = np.finfo(np.float64).eps
    rtol = 1e-07
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        for out_dt in IUINT_TYPES:
            kinds = np.dtype(in_dt).kind + np.dtype(out_dt).kind
            if kinds in ('ii', 'uu', 'ui'):
                arrs = (np.array([iinf.min, iinf.max], dtype=in_dt),)
            elif kinds == 'iu':
                arrs = (np.array([iinf.min, 0], dtype=in_dt), np.array([0, iinf.max], dtype=in_dt))
            for arr in arrs:
                try:
                    aw = SlopeArrayWriter(arr, out_dt)
                except ScalingError:
                    continue
                assert not aw.slope == 0
                arr_back_sc = round_trip(aw)
                adiff = int_abs(arr - arr_back_sc)
                rdiff = adiff / (arr + eps)
                assert np.all(rdiff < rtol)