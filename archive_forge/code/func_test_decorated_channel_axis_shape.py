import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@testing.parametrize('channel_axis', [None, 0, 1, 2, -1, -2, -3])
def test_decorated_channel_axis_shape(channel_axis):
    x = np.zeros((2, 3, 4))
    size = _decorated_channel_axis_size(x, channel_axis=channel_axis)
    if channel_axis is None:
        assert size is None
    else:
        assert size == x.shape[channel_axis]