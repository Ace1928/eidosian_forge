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
@pytest.mark.parametrize('fmt', ['png', 'jpg', 'jpeg', 'tiff'])
def test_imsave(fmt):
    has_alpha = fmt not in ['jpg', 'jpeg']
    np.random.seed(1)
    data = np.random.rand(1856, 2)
    buff_dpi1 = io.BytesIO()
    plt.imsave(buff_dpi1, data, format=fmt, dpi=1)
    buff_dpi100 = io.BytesIO()
    plt.imsave(buff_dpi100, data, format=fmt, dpi=100)
    buff_dpi1.seek(0)
    arr_dpi1 = plt.imread(buff_dpi1, format=fmt)
    buff_dpi100.seek(0)
    arr_dpi100 = plt.imread(buff_dpi100, format=fmt)
    assert arr_dpi1.shape == (1856, 2, 3 + has_alpha)
    assert arr_dpi100.shape == (1856, 2, 3 + has_alpha)
    assert_array_equal(arr_dpi1, arr_dpi100)