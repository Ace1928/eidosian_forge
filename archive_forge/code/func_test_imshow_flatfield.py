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
@image_comparison(['imshow_flatfield.png'], remove_text=True, style='mpl20')
def test_imshow_flatfield():
    fig, ax = plt.subplots()
    im = ax.imshow(np.ones((5, 5)), interpolation='nearest')
    im.set_clim(0.5, 1.5)