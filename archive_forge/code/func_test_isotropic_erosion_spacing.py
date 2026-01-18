import numpy as np
from numpy.testing import assert_array_equal
from skimage import color, data, morphology
from skimage.morphology import binary, isotropic
from skimage.util import img_as_bool
def test_isotropic_erosion_spacing():
    isotropic_res = isotropic.isotropic_dilation(bw_img, 6, spacing=(1, 2))
    binary_res = img_as_bool(binary.binary_dilation(bw_img, _disk_with_spacing(6, spacing=(1, 2))))
    assert_array_equal(isotropic_res, binary_res)