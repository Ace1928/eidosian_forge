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
def test_get_window_extent_for_AxisImage():
    im = np.array([[0.25, 0.75, 1.0, 0.75], [0.1, 0.65, 0.5, 0.4], [0.6, 0.3, 0.0, 0.2], [0.7, 0.9, 0.4, 0.6]])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest')
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)
    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(1, 2)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest', transform=ax.transAxes)
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)
    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])