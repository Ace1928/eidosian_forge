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
def test__resample_valid_output():
    resample = functools.partial(mpl._image.resample, transform=Affine2D())
    with pytest.raises(ValueError, match='must be a NumPy array'):
        resample(np.zeros((9, 9)), None)
    with pytest.raises(ValueError, match='different dimensionalities'):
        resample(np.zeros((9, 9)), np.zeros((9, 9, 4)))
    with pytest.raises(ValueError, match='must be RGBA'):
        resample(np.zeros((9, 9, 4)), np.zeros((9, 9, 3)))
    with pytest.raises(ValueError, match='Mismatched types'):
        resample(np.zeros((9, 9), np.uint8), np.zeros((9, 9)))
    with pytest.raises(ValueError, match='must be C-contiguous'):
        resample(np.zeros((9, 9)), np.zeros((9, 9)).T)