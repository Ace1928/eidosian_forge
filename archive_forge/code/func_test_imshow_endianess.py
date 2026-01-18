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
@image_comparison(['imshow_endianess.png'], remove_text=True)
def test_imshow_endianess():
    x = np.arange(10)
    X, Y = np.meshgrid(x, x)
    Z = np.hypot(X - 5, Y - 5)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    kwargs = dict(origin='lower', interpolation='nearest', cmap='viridis')
    ax1.imshow(Z.astype('<f8'), **kwargs)
    ax2.imshow(Z.astype('>f8'), **kwargs)