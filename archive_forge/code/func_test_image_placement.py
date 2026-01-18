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
@image_comparison(['image_placement'], extensions=['svg', 'pdf'], remove_text=True, style='mpl20')
def test_image_placement():
    """
    The red box should line up exactly with the outside of the image.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color='r', lw=0.1)
    np.random.seed(19680801)
    ax.imshow(np.random.randn(16, 16), cmap='Blues', extent=(0, 1, 0, 1), interpolation='none', vmin=-1, vmax=1)
    ax.set_xlim(-0.1, 1 + 0.1)
    ax.set_ylim(-0.1, 1 + 0.1)