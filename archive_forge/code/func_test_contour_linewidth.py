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
@pytest.mark.parametrize('rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected', [(1.23, None, None, 1.23), (1.23, 4.24, None, 4.24), (1.23, 4.24, 5.02, 5.02)])
def test_contour_linewidth(rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected):
    with rc_context(rc={'lines.linewidth': rc_lines_linewidth, 'contour.linewidth': rc_contour_linewidth}):
        fig, ax = plt.subplots()
        X = np.arange(4 * 3).reshape(4, 3)
        cs = ax.contour(X, linewidths=call_linewidths)
        assert cs.get_linewidths()[0] == expected
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='tlinewidths'):
            assert cs.tlinewidths[0][0] == expected