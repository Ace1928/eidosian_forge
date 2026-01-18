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
@image_comparison(['imshow_bignumbers_real.png'], remove_text=True, style='mpl20')
def test_imshow_bignumbers_real():
    rcParams['image.interpolation'] = 'nearest'
    fig, ax = plt.subplots()
    img = np.array([[2.0, 1.0, 1e+22], [4.0, 1.0, 3.0]])
    pc = ax.imshow(img)
    pc.set_clim(0, 5)