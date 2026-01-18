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
@pytest.mark.parametrize('toolbar', ['none', 'toolbar2', 'toolmanager'])
def test_TextBox(ax, toolbar):
    plt.rcParams._set('toolbar', toolbar)
    submit_event = mock.Mock(spec=noop, return_value=None)
    text_change_event = mock.Mock(spec=noop, return_value=None)
    tool = widgets.TextBox(ax, '')
    tool.on_submit(submit_event)
    tool.on_text_change(text_change_event)
    assert tool.text == ''
    do_event(tool, '_click')
    tool.set_val('x**2')
    assert tool.text == 'x**2'
    assert text_change_event.call_count == 1
    tool.begin_typing()
    tool.stop_typing()
    assert submit_event.call_count == 2
    do_event(tool, '_click', xdata=0.5, ydata=0.5)
    do_event(tool, '_keypress', key='+')
    do_event(tool, '_keypress', key='5')
    assert text_change_event.call_count == 3