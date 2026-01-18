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
def test_SymLogNorm():
    """
    Test SymLogNorm behavior
    """
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2, base=np.e)
    vals = np.array([-30, -1, 2, 6], dtype=float)
    normed_vals = norm(vals)
    expected = [0.0, 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2, base=np.e)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)
    norm = mcolors.SymLogNorm(1, vmin=-np.e ** 3, vmax=np.e ** 3, base=np.e)
    nn = norm([-np.e ** 3, -np.e ** 2, -np.e ** 1, -1, 0, 1, np.e ** 1, np.e ** 2, np.e ** 3])
    xx = np.array([0.0, 0.109123, 0.218246, 0.32737, 0.5, 0.67263, 0.781754, 0.890877, 1.0])
    assert_array_almost_equal(nn, xx)
    norm = mcolors.SymLogNorm(1, vmin=-10 ** 3, vmax=10 ** 3, base=10)
    nn = norm([-10 ** 3, -10 ** 2, -10 ** 1, -1, 0, 1, 10 ** 1, 10 ** 2, 10 ** 3])
    xx = np.array([0.0, 0.121622, 0.243243, 0.364865, 0.5, 0.635135, 0.756757, 0.878378, 1.0])
    assert_array_almost_equal(nn, xx)