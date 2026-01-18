import numpy as np
from numpy.testing import assert_array_equal
from skimage import color, data, morphology
from skimage.morphology import binary, isotropic
from skimage.util import img_as_bool
def test_footprint_overflow():
    img = np.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    isotropic_res = isotropic.isotropic_erosion(img, 9)
    binary_res = img_as_bool(binary.binary_erosion(img, morphology.disk(9)))
    assert_array_equal(isotropic_res, binary_res)