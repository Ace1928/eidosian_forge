import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def check_tick1_visible(axs, x_visible, y_visible):
    """
    Check that the x and y tick visibility is as specified.

    Note: This only checks the tick1line, i.e. bottom / left ticks.
    """
    for ax, visible in zip(axs, x_visible):
        for tick in ax.xaxis.get_major_ticks():
            assert tick.tick1line.get_visible() == visible
    for ax, y_visible in zip(axs, y_visible):
        for tick in ax.yaxis.get_major_ticks():
            assert tick.tick1line.get_visible() == visible