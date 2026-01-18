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
@pytest.mark.parametrize('data, text', [([[10001, 10000]], '[10001.000]'), ([[0.123, 0.987]], '[0.123]'), ([[np.nan, 1, 2]], '[]'), ([[1, 1 + 1e-15]], '[1.0000000000000000]'), ([[-1, -1]], '[-1.0000000000000000]')])
def test_format_cursor_data(data, text):
    from matplotlib.backend_bases import MouseEvent
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    xdisp, ydisp = ax.transData.transform([0, 0])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.format_cursor_data(im.get_cursor_data(event)) == text