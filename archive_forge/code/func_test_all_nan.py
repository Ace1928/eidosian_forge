import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
def test_all_nan():
    x = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    assert_array_almost_equal(plt.contour(x).levels, [-1e-13, -7.5e-14, -5e-14, -2.4e-14, 0.0, 2.4e-14, 5e-14, 7.5e-14, 1e-13])