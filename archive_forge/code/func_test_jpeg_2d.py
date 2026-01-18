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
def test_jpeg_2d():
    imd = np.ones((10, 10), dtype='uint8')
    for i in range(10):
        imd[i, :] = np.linspace(0.0, 1.0, 10) * 255
    im = Image.new('L', (10, 10))
    im.putdata(imd.flatten())
    fig, ax = plt.subplots()
    ax.imshow(im)