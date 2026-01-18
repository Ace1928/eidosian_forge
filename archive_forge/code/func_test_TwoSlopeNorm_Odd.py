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
def test_TwoSlopeNorm_Odd():
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=5)
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert_array_equal(norm(vals), expected)