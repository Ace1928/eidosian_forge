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
def test_cn():
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['blue', 'r'])
    assert mcolors.to_hex('C0') == '#0000ff'
    assert mcolors.to_hex('C1') == '#ff0000'
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['xkcd:blue', 'r'])
    assert mcolors.to_hex('C0') == '#0343df'
    assert mcolors.to_hex('C1') == '#ff0000'
    assert mcolors.to_hex('C10') == '#0343df'
    assert mcolors.to_hex('C11') == '#ff0000'
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['8e4585', 'r'])
    assert mcolors.to_hex('C0') == '#8e4585'
    assert mcolors.to_rgb('C0')[0] != np.inf