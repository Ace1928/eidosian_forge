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
@pytest.mark.parametrize('make_norm', [colors.Normalize, colors.LogNorm, lambda: colors.SymLogNorm(1), lambda: colors.PowerNorm(1)])
def test_empty_imshow(make_norm):
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match='Attempting to set identical low and high xlims'):
        im = ax.imshow([[]], norm=make_norm())
    im.set_extent([-5, 5, -5, 5])
    fig.canvas.draw()
    with pytest.raises(RuntimeError):
        im.make_image(fig.canvas.get_renderer())