import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage._shared._warnings import expected_warnings
from skimage.feature import peak
def test_exclude_border_errors():
    image = np.zeros((5, 5))
    with pytest.raises(ValueError):
        assert peak.peak_local_max(image, exclude_border=(1,))
    with pytest.raises(TypeError):
        assert peak.peak_local_max(image, exclude_border=1.0)
    with pytest.raises(ValueError):
        assert peak.peak_local_max(image, exclude_border=(1, 'a'))
    with pytest.raises(ValueError):
        assert peak.peak_local_max(image, exclude_border=(1, -1))
    with pytest.raises(ValueError):
        assert peak.peak_local_max(image, exclude_border=-1)