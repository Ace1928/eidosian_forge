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
def test_cursor_data():
    from matplotlib.backend_bases import MouseEvent
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='upper')
    x, y = (4, 4)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44
    x, y = (10.1, 4)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None
    ax.clear()
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='lower')
    x, y = (4, 4)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 0.5, 0, 0.5])
    x, y = (0.25, 0.25)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 55
    x, y = (0.75, 0.25)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None
    x, y = (0.01, -0.01)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None
    trans = Affine2D().scale(2).rotate(0.5)
    im = ax.imshow(np.arange(100).reshape(10, 10), transform=trans + ax.transData)
    x, y = (3, 10)
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44