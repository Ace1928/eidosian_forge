import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
@pytest.mark.parametrize('kwargs', [dict(), dict(useblit=False, props=dict(color='red')), dict(useblit=True, button=1)])
def test_lasso_selector(ax, kwargs):
    onselect = mock.Mock(spec=noop, return_value=None)
    tool = widgets.LassoSelector(ax, onselect, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=125, button=1)
    do_event(tool, 'release', xdata=150, ydata=150, button=1)
    onselect.assert_called_once_with([(100, 100), (125, 125), (150, 150)])