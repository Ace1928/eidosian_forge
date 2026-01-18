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
@mpl.style.context('mpl20')
def test_exact_vmin():
    cmap = copy(mpl.colormaps['autumn_r'])
    cmap.set_under(color='lightgrey')
    fig = plt.figure(figsize=(1.9, 0.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    data = np.array([[-1, -1, -1, 0, 0, 0, 0, 43, 79, 95, 66, 1, -1, -1, -1, 0, 0, 0, 34]], dtype=float)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.axis('off')
    fig.canvas.draw()
    from_image = im.make_image(fig.canvas.renderer)[0][0]
    direct_computation = (im.cmap(im.norm((data * ([[1]] * 10)).T.ravel())) * 255).astype(int)
    assert np.all(from_image == direct_computation)