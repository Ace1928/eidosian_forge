import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
@pytest.mark.parametrize('method', ['autolevel', 'equalize', 'gradient', 'maximum', 'mean', 'subtract_mean', 'median', 'minimum', 'modal', 'enhance_contrast', 'pop', 'threshold'])
def test_compare_8bit_vs_16bit(self, method):
    image8 = util.img_as_ubyte(data.camera())[::2, ::2]
    image16 = image8.astype(np.uint16)
    assert_equal(image8, image16)
    np.random.seed(0)
    volume8 = np.random.randint(128, high=256, size=(10, 10, 10), dtype=np.uint8)
    volume16 = volume8.astype(np.uint16)
    methods_3d = ['equalize', 'otsu', 'autolevel', 'gradient', 'majority', 'maximum', 'mean', 'geometric_mean', 'subtract_mean', 'median', 'minimum', 'modal', 'enhance_contrast', 'pop', 'sum', 'threshold', 'noise_filter', 'entropy']
    func = getattr(rank, method)
    f8 = func(image8, disk(3))
    f16 = func(image16, disk(3))
    assert_equal(f8, f16)
    if method in methods_3d:
        f8 = func(volume8, ball(3))
        f16 = func(volume16, ball(3))
        assert_equal(f8, f16)