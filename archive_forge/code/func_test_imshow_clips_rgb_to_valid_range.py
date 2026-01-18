from contextlib import ExitStack
from copy import copy
import functools
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import matplotlib as mpl
from matplotlib import (
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker
import pytest
@pytest.mark.parametrize('dtype', [np.dtype(s) for s in 'u2 u4 i2 i4 i8 f4 f8'.split()])
def test_imshow_clips_rgb_to_valid_range(dtype):
    arr = np.arange(300, dtype=dtype).reshape((10, 10, 3))
    if dtype.kind != 'u':
        arr -= 10
    too_low = arr < 0
    too_high = arr > 255
    if dtype.kind == 'f':
        arr = arr / 255
    _, ax = plt.subplots()
    out = ax.imshow(arr).get_array()
    assert (out[too_low] == 0).all()
    if dtype.kind == 'f':
        assert (out[too_high] == 1).all()
        assert out.dtype.kind == 'f'
    else:
        assert (out[too_high] == 255).all()
        assert out.dtype == np.uint8