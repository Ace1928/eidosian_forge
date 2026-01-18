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
def test_lasso_selector_set_props(ax):
    onselect = mock.Mock(spec=noop, return_value=None)
    tool = widgets.LassoSelector(ax, onselect, props=dict(color='b', alpha=0.2))
    artist = tool._selection_artist
    assert mcolors.same_color(artist.get_color(), 'b')
    assert artist.get_alpha() == 0.2
    tool.set_props(color='r', alpha=0.3)
    assert mcolors.same_color(artist.get_color(), 'r')
    assert artist.get_alpha() == 0.3