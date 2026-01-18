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
def plane(azimuth, elevation, x, y):
    """
        Create a plane whose normal vector is at the given azimuth and
        elevation.
        """
    theta, phi = _azimuth2math(azimuth, elevation)
    a, b, c = _sph2cart(theta, phi)
    z = -(a * x + b * y) / c
    return z