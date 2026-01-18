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
@image_comparison(['bbox_image_inverted'], remove_text=True, style='mpl20')
def test_bbox_image_inverted():
    image = np.arange(100).reshape((10, 10))
    fig, ax = plt.subplots()
    bbox_im = BboxImage(TransformedBbox(Bbox([[100, 100], [0, 0]]), ax.transData), interpolation='nearest')
    bbox_im.set_data(image)
    bbox_im.set_clip_on(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.add_artist(bbox_im)
    image = np.identity(10)
    bbox_im = BboxImage(TransformedBbox(Bbox([[0.1, 0.2], [0.3, 0.25]]), ax.figure.transFigure), interpolation='nearest')
    bbox_im.set_data(image)
    bbox_im.set_clip_on(False)
    ax.add_artist(bbox_im)