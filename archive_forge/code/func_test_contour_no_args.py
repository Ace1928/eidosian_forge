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
def test_contour_no_args():
    fig, ax = plt.subplots()
    data = [[0, 1], [1, 0]]
    with pytest.raises(TypeError, match='contour\\(\\) takes from 1 to 4'):
        ax.contour(Z=data)