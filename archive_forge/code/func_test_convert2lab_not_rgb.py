import numpy as np
import pytest
from skimage.segmentation import quickshift
from skimage._shared import testing
from skimage._shared.testing import (
def test_convert2lab_not_rgb():
    img = np.zeros((20, 21, 2))
    with pytest.raises(ValueError, match='Only RGB images can be converted to Lab space'):
        quickshift(img, convert2lab=True)