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
def test_colormap_alpha_array():
    cmap = mpl.colormaps['viridis']
    vals = [-1, 0.5, 2]
    with pytest.raises(ValueError, match='alpha is array-like but'):
        cmap(vals, alpha=[1, 1, 1, 1])
    alpha = np.array([0.1, 0.2, 0.3])
    c = cmap(vals, alpha=alpha)
    assert_array_equal(c[:, -1], alpha)
    c = cmap(vals, alpha=alpha, bytes=True)
    assert_array_equal(c[:, -1], (alpha * 255).astype(np.uint8))