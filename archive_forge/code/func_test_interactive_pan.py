import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
@pytest.mark.parametrize('key,mouseend,expectedxlim,expectedylim', [(None, (0.2, 0.2), (3.49, 12.49), (2.7, 11.7)), (None, (0.2, 0.5), (3.49, 12.49), (0, 9)), (None, (0.5, 0.2), (0, 9), (2.7, 11.7)), (None, (0.5, 0.5), (0, 9), (0, 9)), (None, (0.8, 0.25), (-3.47, 5.53), (2.25, 11.25)), (None, (0.2, 0.25), (3.49, 12.49), (2.25, 11.25)), (None, (0.8, 0.85), (-3.47, 5.53), (-3.14, 5.86)), (None, (0.2, 0.85), (3.49, 12.49), (-3.14, 5.86)), ('shift', (0.2, 0.4), (3.49, 12.49), (0, 9)), ('shift', (0.4, 0.2), (0, 9), (2.7, 11.7)), ('shift', (0.2, 0.25), (3.49, 12.49), (3.49, 12.49)), ('shift', (0.8, 0.25), (-3.47, 5.53), (3.47, 12.47)), ('shift', (0.8, 0.9), (-3.58, 5.41), (-3.58, 5.41)), ('shift', (0.2, 0.85), (3.49, 12.49), (-3.49, 5.51)), ('x', (0.2, 0.1), (3.49, 12.49), (0, 9)), ('y', (0.1, 0.2), (0, 9), (2.7, 11.7)), ('control', (0.2, 0.2), (3.49, 12.49), (3.49, 12.49)), ('control', (0.4, 0.2), (2.72, 11.72), (2.72, 11.72))])
def test_interactive_pan(key, mouseend, expectedxlim, expectedylim):
    fig, ax = plt.subplots()
    ax.plot(np.arange(10))
    assert ax.get_navigate()
    ax.set_aspect('equal')
    mousestart = (0.5, 0.5)
    sstart = ax.transData.transform(mousestart).astype(int)
    send = ax.transData.transform(mouseend).astype(int)
    start_event = MouseEvent('button_press_event', fig.canvas, *sstart, button=MouseButton.LEFT, key=key)
    stop_event = MouseEvent('button_release_event', fig.canvas, *send, button=MouseButton.LEFT, key=key)
    tb = NavigationToolbar2(fig.canvas)
    tb.pan()
    tb.press_pan(start_event)
    tb.drag_pan(stop_event)
    tb.release_pan(stop_event)
    assert tuple(ax.get_xlim()) == pytest.approx(expectedxlim, abs=0.02)
    assert tuple(ax.get_ylim()) == pytest.approx(expectedylim, abs=0.02)