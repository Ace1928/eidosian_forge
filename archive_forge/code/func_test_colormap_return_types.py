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
def test_colormap_return_types():
    """
    Make sure that tuples are returned for scalar input and
    that the proper shapes are returned for ndarrays.
    """
    cmap = mpl.colormaps['plasma']
    assert isinstance(cmap(0.5), tuple)
    assert len(cmap(0.5)) == 4
    x = np.ones(4)
    assert cmap(x).shape == x.shape + (4,)
    x2d = np.zeros((2, 2))
    assert cmap(x2d).shape == x2d.shape + (4,)