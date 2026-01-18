import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@_isolated_tk_test(success_count=2)
def test_missing_back_button():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

    class Toolbar(NavigationToolbar2Tk):
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] in ('Home', 'Pan', 'Zoom')]
    fig = plt.figure()
    print('success')
    Toolbar(fig.canvas, fig.canvas.manager.window)
    print('success')