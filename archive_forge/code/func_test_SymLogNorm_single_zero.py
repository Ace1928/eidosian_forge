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
def test_SymLogNorm_single_zero():
    """
    Test SymLogNorm to ensure it is not adding sub-ticks to zero label
    """
    fig = plt.figure()
    norm = mcolors.SymLogNorm(1e-05, vmin=-1, vmax=1, base=np.e)
    cbar = mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    ticks = cbar.get_ticks()
    assert np.count_nonzero(ticks == 0) <= 1
    plt.close(fig)