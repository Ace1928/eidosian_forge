import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_distance_transform_cdt_invalid_metric():
    msg = 'invalid metric provided'
    with pytest.raises(ValueError, match=msg):
        ndimage.distance_transform_cdt(np.ones((5, 5)), metric='garbage')