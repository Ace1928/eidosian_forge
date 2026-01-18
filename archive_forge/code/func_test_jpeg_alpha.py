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
def test_jpeg_alpha():
    plt.figure(figsize=(1, 1), dpi=300)
    im = np.zeros((300, 300, 4), dtype=float)
    im[..., 3] = np.linspace(0.0, 1.0, 300)
    plt.figimage(im)
    buff = io.BytesIO()
    plt.savefig(buff, facecolor='red', format='jpg', dpi=300)
    buff.seek(0)
    image = Image.open(buff)
    num_colors = len(image.getcolors(256))
    assert 175 <= num_colors <= 210
    corner_pixel = image.getpixel((0, 0))
    assert corner_pixel == (254, 0, 0)