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
@image_comparison(['mask_image'], remove_text=True)
def test_mask_image():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    A = np.ones((5, 5))
    A[1:2, 1:2] = np.nan
    ax1.imshow(A, interpolation='nearest')
    A = np.zeros((5, 5), dtype=bool)
    A[1:2, 1:2] = True
    A = np.ma.masked_array(np.ones((5, 5), dtype=np.uint16), A)
    ax2.imshow(A, interpolation='nearest')