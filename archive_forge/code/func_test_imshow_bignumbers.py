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
@image_comparison(['imshow_bignumbers.png'], remove_text=True, style='mpl20')
def test_imshow_bignumbers():
    rcParams['image.interpolation'] = 'nearest'
    fig, ax = plt.subplots()
    img = np.array([[1, 2, 1000000000000.0], [3, 1, 4]], dtype=np.uint64)
    pc = ax.imshow(img)
    pc.set_clim(0, 5)