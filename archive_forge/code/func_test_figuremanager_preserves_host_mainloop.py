import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
@_isolated_tk_test(success_count=1)
def test_figuremanager_preserves_host_mainloop():
    import tkinter
    import matplotlib.pyplot as plt
    success = []

    def do_plot():
        plt.figure()
        plt.plot([1, 2], [3, 5])
        plt.close()
        root.after(0, legitimate_quit)

    def legitimate_quit():
        root.quit()
        success.append(True)
    root = tkinter.Tk()
    root.after(0, do_plot)
    root.mainloop()
    if success:
        print('success')