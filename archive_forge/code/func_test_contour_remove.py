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
def test_contour_remove():
    ax = plt.figure().add_subplot()
    orig_children = ax.get_children()
    cs = ax.contour(np.arange(16).reshape((4, 4)))
    cs.clabel()
    assert ax.get_children() != orig_children
    cs.remove()
    assert ax.get_children() == orig_children