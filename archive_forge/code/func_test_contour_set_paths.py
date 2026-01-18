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
@check_figures_equal(extensions=['png'])
def test_contour_set_paths(fig_test, fig_ref):
    cs_test = fig_test.subplots().contour([[0, 1], [1, 2]])
    cs_ref = fig_ref.subplots().contour([[1, 0], [2, 1]])
    cs_test.set_paths(cs_ref.get_paths())