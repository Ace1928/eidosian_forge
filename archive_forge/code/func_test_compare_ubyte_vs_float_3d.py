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
def test_compare_ubyte_vs_float_3d(self):
    np.random.seed(0)
    volume_uint = np.random.randint(0, high=256, size=(10, 20, 30), dtype=np.uint8)
    volume_float = img_as_float(volume_uint)
    methods_3d = ['equalize', 'otsu', 'autolevel', 'gradient', 'majority', 'maximum', 'mean', 'geometric_mean', 'subtract_mean', 'median', 'minimum', 'modal', 'enhance_contrast', 'pop', 'sum', 'threshold', 'noise_filter', 'entropy']
    for method in methods_3d:
        func = getattr(rank, method)
        out_u = func(volume_uint, ball(3))
        with expected_warnings(['Possible precision loss']):
            out_f = func(volume_float, ball(3))
        assert_equal(out_u, out_f)