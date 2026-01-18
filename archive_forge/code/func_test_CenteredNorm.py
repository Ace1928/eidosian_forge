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
def test_CenteredNorm():
    np.random.seed(0)
    x = np.random.normal(size=100)
    x_maxabs = np.max(np.abs(x))
    norm_ref = mcolors.Normalize(vmin=-x_maxabs, vmax=x_maxabs)
    norm = mcolors.CenteredNorm()
    assert_array_almost_equal(norm_ref(x), norm(x))
    vcenter = int(np.random.normal(scale=50))
    norm = mcolors.CenteredNorm(vcenter=vcenter)
    norm.autoscale_None([1, 2])
    assert norm.vmax + norm.vmin == 2 * vcenter
    norm = mcolors.CenteredNorm(halfrange=1.0)
    norm.autoscale_None([1, 3000])
    assert norm.halfrange == 1.0
    x = np.random.normal(size=10)
    norm = mcolors.CenteredNorm(vcenter=0.5, halfrange=0.5)
    assert_array_almost_equal(x, norm(x))
    norm = mcolors.CenteredNorm(vcenter=1, halfrange=1)
    assert_array_almost_equal(x, 2 * norm(x))
    norm = mcolors.CenteredNorm()
    norm.vcenter = 2
    norm.halfrange = 2
    assert_array_almost_equal(x, 4 * norm(x))
    norm = mcolors.CenteredNorm()
    norm.halfrange = 2
    norm.vcenter = 2
    assert_array_almost_equal(x, 4 * norm(x))
    norm = mcolors.CenteredNorm()
    assert norm.vcenter == 0
    norm(np.linspace(-1.0, 0.0, 10))
    assert norm.vmax == 1.0
    assert norm.halfrange == 1.0
    norm.vcenter = 1
    assert norm.vmin == 0
    assert norm.vmax == 2
    assert norm.halfrange == 1
    norm.vmin = -1
    assert norm.halfrange == 2
    assert norm.vmax == 3
    assert norm.vcenter == 1
    norm.vmax = 2
    assert norm.halfrange == 1
    assert norm.vmin == 0
    assert norm.vcenter == 1