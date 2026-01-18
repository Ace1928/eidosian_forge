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
def test_contour_Nlevels():
    z = np.arange(12).reshape((3, 4))
    fig, ax = plt.subplots()
    cs1 = ax.contour(z, 5)
    assert len(cs1.levels) > 1
    cs2 = ax.contour(z, levels=5)
    assert (cs1.levels == cs2.levels).all()