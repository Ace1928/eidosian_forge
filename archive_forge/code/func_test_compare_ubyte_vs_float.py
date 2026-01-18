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
def test_compare_ubyte_vs_float(self):
    image_uint = img_as_ubyte(data.camera()[:50, :50])
    image_float = img_as_float(image_uint)
    methods = ['autolevel', 'equalize', 'gradient', 'threshold', 'subtract_mean', 'enhance_contrast', 'pop']
    for method in methods:
        func = getattr(rank, method)
        out_u = func(image_uint, disk(3))
        with expected_warnings(['Possible precision loss']):
            out_f = func(image_float, disk(3))
        assert_equal(out_u, out_f)