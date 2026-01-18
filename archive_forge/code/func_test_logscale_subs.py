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
def test_logscale_subs():
    fig, ax = plt.subplots()
    ax.set_yscale('log', subs=np.array([2, 3, 4]))
    fig.canvas.draw()