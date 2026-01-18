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
def test_compare_with_gray_erosion(self):
    image = (np.random.rand(100, 100) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    for r in range(3, 20, 2):
        elem = np.ones((r, r), dtype=np.uint8)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask)
        cm = gray.erosion(image, elem)
        assert_equal(out, cm)