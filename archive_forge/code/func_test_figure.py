import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
def test_figure(master):
    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([1, 2, 3])
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.draw()
    canvas.mpl_connect('key_press_event', key_press_handler)
    canvas.get_tk_widget().pack(expand=True, fill='both')
    toolbar = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
    toolbar.pack(expand=True, fill='x')
    canvas.get_tk_widget().forget()
    toolbar.forget()