import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_gifti_image():
    gi = GiftiImage()
    assert gi.darrays == []
    assert gi.meta == {}
    assert gi.labeltable.labels == []
    arr = np.zeros((2, 3))
    gi.darrays.append(arr)
    gi = GiftiImage()
    assert gi.darrays == []
    gi = GiftiImage()
    assert gi.numDA == 0
    data = rng.random(5, dtype=np.float32)
    da = GiftiDataArray(data)
    gi.add_gifti_data_array(da)
    assert gi.numDA == 1
    assert_array_equal(gi.darrays[0].data, data)
    gi.remove_gifti_data_array(0)
    assert gi.numDA == 0
    gi = GiftiImage()
    gi.remove_gifti_data_array_by_intent(0)
    assert gi.numDA == 0
    gi = GiftiImage()
    da = GiftiDataArray(np.zeros((5,), np.float32), intent=0)
    gi.add_gifti_data_array(da)
    gi.remove_gifti_data_array_by_intent(3)
    assert gi.numDA == 1, "data array should exist on 'missed' remove"
    gi.remove_gifti_data_array_by_intent(da.intent)
    assert gi.numDA == 0