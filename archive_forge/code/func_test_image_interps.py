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
@image_comparison(['image_interps'], style='mpl20')
def test_image_interps():
    """Make the basic nearest, bilinear and bicubic interps."""
    plt.rcParams['text.kerning_factor'] = 6
    X = np.arange(100).reshape(5, 20)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.imshow(X, interpolation='nearest')
    ax1.set_title('three interpolations')
    ax1.set_ylabel('nearest')
    ax2.imshow(X, interpolation='bilinear')
    ax2.set_ylabel('bilinear')
    ax3.imshow(X, interpolation='bicubic')
    ax3.set_ylabel('bicubic')