import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_canvas_reinit():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        fig.canvas.draw_idle()
        called = True
    fig, ax = plt.subplots()
    fig.stale_callback = crashing_callback
    canvas = FigureCanvasQTAgg(fig)
    fig.stale = True
    assert called