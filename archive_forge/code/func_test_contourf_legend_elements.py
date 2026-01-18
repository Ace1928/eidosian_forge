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
def test_contourf_legend_elements():
    from matplotlib.patches import Rectangle
    x = np.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y
    cs = plt.contourf(h, levels=[10, 30, 50], colors=['#FFFF00', '#FF00FF', '#00FFFF'], extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    artists, labels = cs.legend_elements()
    assert labels == ['$x \\leq -1e+250s$', '$10.0 < x \\leq 30.0$', '$30.0 < x \\leq 50.0$', '$x > 1e+250s$']
    expected_colors = ('blue', '#FFFF00', '#FF00FF', 'red')
    assert all((isinstance(a, Rectangle) for a in artists))
    assert all((same_color(a.get_facecolor(), c) for a, c in zip(artists, expected_colors)))