import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_axes_limit_invalid():
    data = np.empty((50, 50))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 51)], axes=[0])