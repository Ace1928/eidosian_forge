import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_invalid_footprint():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((4, 4)))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((3, 4)))
    reconstruction(seed, mask, footprint=np.ones((3, 3)))