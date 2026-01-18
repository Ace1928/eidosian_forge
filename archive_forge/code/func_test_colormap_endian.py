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
def test_colormap_endian():
    """
    GitHub issue #1005: a bug in putmask caused erroneous
    mapping of 1.0 when input from a non-native-byteorder
    array.
    """
    cmap = mpl.colormaps['jet']
    a = [-0.5, 0, 0.5, 1, 1.5, np.nan]
    for dt in ['f2', 'f4', 'f8']:
        anative = np.ma.masked_invalid(np.array(a, dtype=dt))
        aforeign = anative.byteswap().view(anative.dtype.newbyteorder())
        assert_array_equal(cmap(anative), cmap(aforeign))