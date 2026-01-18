import numpy as np
import pytest
from skimage.measure import (
def test_moc():
    img1 = np.ones((4, 4))
    img2 = 2 * np.ones((4, 4))
    assert manders_overlap_coeff(img1, img2) == 1
    img_negativeint = np.where(img1 == 1, -1, img1)
    img_negativefloat = img_negativeint / 2.0
    with pytest.raises(ValueError):
        manders_overlap_coeff(img_negativeint, img2)
    with pytest.raises(ValueError):
        manders_overlap_coeff(img1, img_negativeint)
    with pytest.raises(ValueError):
        manders_overlap_coeff(img_negativefloat, img2)
    with pytest.raises(ValueError):
        manders_overlap_coeff(img1, img_negativefloat)
    with pytest.raises(ValueError):
        manders_overlap_coeff(img_negativefloat, img_negativefloat)