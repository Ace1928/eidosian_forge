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
def test_colormap_bad_data_with_alpha():
    cmap = mpl.colormaps['viridis']
    c = cmap(np.nan, alpha=0.5)
    assert c == (0, 0, 0, 0)
    c = cmap([0.5, np.nan], alpha=0.5)
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([0.5, np.nan], alpha=[0.1, 0.2])
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=0.5)
    assert_array_equal(c[0, 0], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=np.full((2, 2), 0.5))
    assert_array_equal(c[0, 0], (0, 0, 0, 0))