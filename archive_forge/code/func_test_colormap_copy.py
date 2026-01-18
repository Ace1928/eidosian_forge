import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_colormap_copy():
    cmap = plt.cm.Reds
    copied_cmap = copy.copy(cmap)
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, 0.5, 1, np.nan, np.inf])
    cmap2 = copy.copy(copied_cmap)
    cmap2.set_bad('g')
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, 0.5, 1, np.nan, np.inf])
    assert_array_equal(ret1, ret2)
    cmap = plt.cm.Reds
    copied_cmap = cmap.copy()
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, 0.5, 1, np.nan, np.inf])
    cmap2 = copy.copy(copied_cmap)
    cmap2.set_bad('g')
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, 0.5, 1, np.nan, np.inf])
    assert_array_equal(ret1, ret2)