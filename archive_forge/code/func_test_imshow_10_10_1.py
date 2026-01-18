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
@check_figures_equal(extensions=['png'])
def test_imshow_10_10_1(fig_test, fig_ref):
    arr = np.arange(100).reshape((10, 10, 1))
    ax = fig_ref.subplots()
    ax.imshow(arr[:, :, 0], interpolation='bilinear', extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax = fig_test.subplots()
    ax.imshow(arr, interpolation='bilinear', extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)