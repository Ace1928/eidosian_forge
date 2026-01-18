import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
@pytest.mark.parametrize('plot_func', ['imshow', 'contourf'])
@pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
@pytest.mark.parametrize('tool,button,expected', [('zoom', MouseButton.LEFT, (4, 6)), ('zoom', MouseButton.RIGHT, (-20, 30)), ('pan', MouseButton.LEFT, (-2, 8)), ('pan', MouseButton.RIGHT, (1.47, 7.78))])
def test_interactive_colorbar(plot_func, orientation, tool, button, expected):
    fig, ax = plt.subplots()
    data = np.arange(12).reshape((4, 3))
    vmin0, vmax0 = (0, 10)
    coll = getattr(ax, plot_func)(data, vmin=vmin0, vmax=vmax0)
    cb = fig.colorbar(coll, ax=ax, orientation=orientation)
    if plot_func == 'contourf':
        assert not cb.ax.get_navigate()
        return
    assert cb.ax.get_navigate()
    vmin, vmax = (4, 6)
    d0 = (vmin, 0.5)
    d1 = (vmax, 0.5)
    if orientation == 'vertical':
        d0 = d0[::-1]
        d1 = d1[::-1]
    s0 = cb.ax.transData.transform(d0).astype(int)
    s1 = cb.ax.transData.transform(d1).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *s0, button)
    stop_event = MouseEvent('button_release_event', fig.canvas, *s1, button)
    tb = NavigationToolbar2(fig.canvas)
    if tool == 'zoom':
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)
    assert (cb.vmin, cb.vmax) == pytest.approx(expected, abs=0.15)