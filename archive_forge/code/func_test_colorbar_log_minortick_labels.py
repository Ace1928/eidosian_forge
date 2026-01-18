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
def test_colorbar_log_minortick_labels():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        pcm = ax.imshow([[10000, 50000]], norm=LogNorm())
        cb = fig.colorbar(pcm)
        fig.canvas.draw()
        lb = [l.get_text() for l in cb.ax.yaxis.get_ticklabels(which='both')]
        expected = ['$\\mathdefault{10^{4}}$', '$\\mathdefault{2\\times10^{4}}$', '$\\mathdefault{3\\times10^{4}}$', '$\\mathdefault{4\\times10^{4}}$']
        for exp in expected:
            assert exp in lb