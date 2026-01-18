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
def test_bitdepth(self):
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty((100, 100), dtype=np.uint16)
    mask = np.ones((100, 100), dtype=np.uint8)
    for i in range(8, 13):
        max_val = 2 ** i - 1
        image = np.full((100, 100), max_val, dtype=np.uint16)
        if i > 10:
            expected = ['Bad rank filter performance']
        else:
            expected = []
        with expected_warnings(expected):
            rank.mean_percentile(image=image, footprint=elem, mask=mask, out=out, shift_x=0, shift_y=0, p0=0.1, p1=0.9)