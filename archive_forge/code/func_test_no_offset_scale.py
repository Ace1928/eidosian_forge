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
def test_no_offset_scale():
    SAW = SlopeArrayWriter
    for data in ((-128, 127), (-128, 126), (-128, -127), (-128, 0), (-128, -1), (126, 127), (-127, 127)):
        aw = SAW(np.array(data, dtype=np.float32), np.int8)
        assert aw.slope == 1.0
    aw = SAW(np.array([-126, 127 * 2.0], dtype=np.float32), np.int8)
    assert aw.slope == 2
    aw = SAW(np.array([-128 * 2.0, 127], dtype=np.float32), np.int8)
    assert aw.slope == 2
    n = -2 ** 15
    aw = SAW(np.array([n, n], dtype=np.int16), np.uint8)
    assert_array_almost_equal(aw.slope, n / 255.0, 5)