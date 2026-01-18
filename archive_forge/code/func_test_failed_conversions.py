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
def test_failed_conversions():
    with pytest.raises(ValueError):
        mcolors.to_rgba('5')
    with pytest.raises(ValueError):
        mcolors.to_rgba('-1')
    with pytest.raises(ValueError):
        mcolors.to_rgba('nan')
    with pytest.raises(ValueError):
        mcolors.to_rgba('unknown_color')
    with pytest.raises(ValueError):
        mcolors.to_rgba(0.4)