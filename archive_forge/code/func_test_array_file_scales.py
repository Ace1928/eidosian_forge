import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale
@pytest.mark.parametrize('in_type, out_type', [(np.int16, np.int16), (np.int16, np.int8), (np.uint16, np.uint8), (np.int32, np.int8), (np.float32, np.uint8), (np.float32, np.int16)])
def test_array_file_scales(in_type, out_type):
    bio = BytesIO()
    out_dtype = np.dtype(out_type)
    arr = np.zeros((3,), dtype=in_type)
    info = type_info(in_type)
    arr[0], arr[1] = (info['min'], info['max'])
    slope, inter, mn, mx = _calculate_scale(arr, out_dtype, True)
    array_to_file(arr, bio, out_type, 0, inter, slope, mn, mx)
    bio.seek(0)
    arr2 = array_from_file(arr.shape, out_dtype, bio)
    arr3 = apply_read_scaling(arr2, slope, inter)
    max_miss = slope / 2.0
    assert np.all(np.abs(arr - arr3) <= max_miss)