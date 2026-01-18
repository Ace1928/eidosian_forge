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
def test_dumber_writers():
    arr = np.arange(10, dtype=np.float64)
    aw = SlopeArrayWriter(arr)
    aw.slope = 2.0
    assert aw.slope == 2.0
    with pytest.raises(AttributeError):
        aw.inter
    aw = ArrayWriter(arr)
    with pytest.raises(AttributeError):
        aw.slope
    with pytest.raises(AttributeError):
        aw.inter
    with pytest.raises(WriterError):
        ArrayWriter(arr, np.int16)