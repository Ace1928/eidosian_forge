import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_log_scatter():
    """Issue #1799"""
    fig, ax = plt.subplots(1)
    x = np.arange(10)
    y = np.arange(10) - 1
    ax.scatter(x, y)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf = io.BytesIO()
    fig.savefig(buf, format='eps')
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')