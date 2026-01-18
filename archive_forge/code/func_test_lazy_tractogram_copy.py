import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
def test_lazy_tractogram_copy(self):
    tractogram = DATA['lazy_tractogram'].copy()
    assert tractogram is not DATA['lazy_tractogram']
    assert tractogram._streamlines is DATA['lazy_tractogram']._streamlines
    assert tractogram._data_per_streamline is not DATA['lazy_tractogram']._data_per_streamline
    assert tractogram._data_per_point is not DATA['lazy_tractogram']._data_per_point
    for key in tractogram.data_per_streamline:
        data = tractogram.data_per_streamline.store[key]
        expected = DATA['lazy_tractogram'].data_per_streamline.store[key]
        assert data is expected
    for key in tractogram.data_per_point:
        data = tractogram.data_per_point.store[key]
        expected = DATA['lazy_tractogram'].data_per_point.store[key]
        assert data is expected
    assert tractogram._affine_to_apply is not DATA['lazy_tractogram']._affine_to_apply
    assert_array_equal(tractogram._affine_to_apply, DATA['lazy_tractogram']._affine_to_apply)
    with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
        assert_tractogram_equal(tractogram, DATA['tractogram'])