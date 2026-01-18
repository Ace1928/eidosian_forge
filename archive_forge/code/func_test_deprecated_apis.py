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
def test_deprecated_apis():
    cs = plt.contour(np.arange(16).reshape((4, 4)))
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='collections'):
        colls = cs.collections
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='tcolors'):
        assert_array_equal(cs.tcolors, [c.get_edgecolor() for c in colls])
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='tlinewidths'):
        assert cs.tlinewidths == [c.get_linewidth() for c in colls]
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='antialiased'):
        assert cs.antialiased
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='antialiased'):
        cs.antialiased = False
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='antialiased'):
        assert not cs.antialiased