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
@image_comparison(['no_interpolation_origin'], remove_text=True)
def test_no_interpolation_origin():
    fig, axs = plt.subplots(2)
    axs[0].imshow(np.arange(100).reshape((2, 50)), origin='lower', interpolation='none')
    axs[1].imshow(np.arange(100).reshape((2, 50)), interpolation='none')