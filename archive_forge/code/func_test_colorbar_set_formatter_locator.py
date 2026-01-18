import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
def test_colorbar_set_formatter_locator():
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(np.random.randn(10, 10))
    cb = fig.colorbar(pc)
    cb.ax.yaxis.set_major_locator(FixedLocator(np.arange(10)))
    cb.ax.yaxis.set_minor_locator(FixedLocator(np.arange(0, 10, 0.2)))
    assert cb.locator is cb.ax.yaxis.get_major_locator()
    assert cb.minorlocator is cb.ax.yaxis.get_minor_locator()
    cb.ax.yaxis.set_major_formatter(LogFormatter())
    cb.ax.yaxis.set_minor_formatter(LogFormatter())
    assert cb.formatter is cb.ax.yaxis.get_major_formatter()
    assert cb.minorformatter is cb.ax.yaxis.get_minor_formatter()
    loc = FixedLocator(np.arange(7))
    cb.locator = loc
    assert cb.ax.yaxis.get_major_locator() is loc
    loc = FixedLocator(np.arange(0, 7, 0.1))
    cb.minorlocator = loc
    assert cb.ax.yaxis.get_minor_locator() is loc
    fmt = LogFormatter()
    cb.formatter = fmt
    assert cb.ax.yaxis.get_major_formatter() is fmt
    fmt = LogFormatter()
    cb.minorformatter = fmt
    assert cb.ax.yaxis.get_minor_formatter() is fmt