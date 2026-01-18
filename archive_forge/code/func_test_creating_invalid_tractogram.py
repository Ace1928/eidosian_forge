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
def test_creating_invalid_tractogram(self):
    scalars = [[(1, 0, 0)] * 1, [(0, 1, 0)] * 2, [(0, 0, 1)] * 3]
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})
    properties = [np.array([1.11, 1.22], dtype='f4'), np.array([3.11, 3.22], dtype='f4')]
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})
    scalars = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})
    properties = [[1.11, 1.22], [2.11], [3.11, 3.22]]
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})
    properties = [np.array([[1.11], [1.22]], dtype='f4'), np.array([[2.11], [2.22]], dtype='f4'), np.array([[3.11], [3.22]], dtype='f4')]
    with pytest.raises(ValueError):
        Tractogram(streamlines=DATA['streamlines'], data_per_streamline={'properties': properties})