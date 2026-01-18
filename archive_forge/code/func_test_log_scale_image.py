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
@image_comparison(['log_scale_image'], remove_text=True)
def test_log_scale_image():
    Z = np.zeros((10, 10))
    Z[::2] = 1
    fig, ax = plt.subplots()
    ax.imshow(Z, extent=[1, 100, 1, 100], cmap='viridis', vmax=1, vmin=-1, aspect='auto')
    ax.set(yscale='log')