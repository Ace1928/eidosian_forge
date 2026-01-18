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
def test_light_source_shading_empty_mask():
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z0 = 10 * np.cos(x ** 2 + y ** 2)
    z1 = np.ma.array(z0)
    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb0 = ls.shade(z0, cmap)
    rgb1 = ls.shade(z1, cmap)
    assert_array_almost_equal(rgb0, rgb1)