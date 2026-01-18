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
def test_arraywriter_check_scaling():
    arr = np.array([0, 1, 128, 255], np.uint8)
    aw = ArrayWriter(arr)
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int8)
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int8, check_scaling=True)
    aw = ArrayWriter(arr, np.int8, check_scaling=False)
    assert_array_equal(round_trip(aw), np.clip(arr, 0, 127))
    with pytest.raises(TypeError):
        ArrayWriter(arr, np.int8, False)