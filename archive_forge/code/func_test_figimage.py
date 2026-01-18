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
@pytest.mark.parametrize('suppressComposite', [False, True])
@image_comparison(['figimage'], extensions=['png', 'pdf'])
def test_figimage(suppressComposite):
    fig = plt.figure(figsize=(2, 2), dpi=100)
    fig.suppressComposite = suppressComposite
    x, y = np.ix_(np.arange(100) / 100.0, np.arange(100) / 100)
    z = np.sin(x ** 2 + y ** 2 - x * y)
    c = np.sin(20 * x ** 2 + 50 * y ** 2)
    img = z + c / 5
    fig.figimage(img, xo=0, yo=0, origin='lower')
    fig.figimage(img[::-1, :], xo=0, yo=100, origin='lower')
    fig.figimage(img[:, ::-1], xo=100, yo=0, origin='lower')
    fig.figimage(img[::-1, ::-1], xo=100, yo=100, origin='lower')