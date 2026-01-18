import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def test_nan2zero_range_ok(self):
    img_class = self.image_class
    arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    arr[0, 0, 0] = np.nan
    arr[1, 0, 0] = 256
    img = img_class(arr, np.eye(4))
    rt_img = bytesio_round_trip(img)
    assert_array_equal(rt_img.get_fdata(), arr)
    img.set_data_dtype(np.uint8)
    with np.errstate(invalid='ignore'):
        rt_img = bytesio_round_trip(img)
    assert rt_img.get_fdata()[0, 0, 0] == 0