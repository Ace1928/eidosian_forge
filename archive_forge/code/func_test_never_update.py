import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@pytest.mark.flaky(reruns=3)
@_isolated_tk_test(success_count=0)
def test_never_update():
    import tkinter
    del tkinter.Misc.update
    del tkinter.Misc.update_idletasks
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.show(block=False)
    plt.draw()
    fig.canvas.toolbar.configure_subplots()
    fig.canvas.get_tk_widget().after(100, plt.close, fig)
    plt.show(block=True)