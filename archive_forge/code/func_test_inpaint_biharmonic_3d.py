import numpy as np
from skimage import data, img_as_float
from skimage._shared import testing
from skimage._shared.testing import assert_allclose
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint
@testing.parametrize('split_into_regions', [False, True])
def test_inpaint_biharmonic_3d(split_into_regions):
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    img = np.dstack((img, img.T))
    mask = np.zeros_like(img)
    mask[2, 2:, :] = 1
    mask[1, 3:, :] = 1
    mask[0, 4:, :] = 1
    img[np.where(mask)] = 0
    out = inpaint.inpaint_biharmonic(img, mask, split_into_regions=split_into_regions)
    ref = np.dstack((np.array([[0.0, 0.0625, 0.25, 0.5625, 0.53752796], [0.0, 0.0625, 0.25, 0.4444378, 0.5376221], [0.0, 0.0625, 0.23693666, 0.46621112, 0.68615592], [0.0, 0.0625, 0.25, 0.5625, 1.0], [0.0, 0.0625, 0.25, 0.5625, 1.0]]), np.array([[0.0, 0.0, 0.0, 0.0, 0.19621902], [0.0625, 0.0625, 0.0625, 0.17470756, 0.30140091], [0.25, 0.25, 0.27241289, 0.3515544, 0.43068654], [0.5625, 0.5625, 0.5625, 0.5625, 0.5625], [1.0, 1.0, 1.0, 1.0, 1.0]])))
    assert_allclose(ref, out)