import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def check_ticklabel_visible(axs, x_visible, y_visible):
    """Check that the x and y ticklabel visibility is as specified."""
    for i, (ax, vx, vy) in enumerate(zip(axs, x_visible, y_visible)):
        for l in ax.get_xticklabels() + [ax.xaxis.offsetText]:
            assert l.get_visible() == vx, f'Visibility of x axis #{i} is incorrectly {vx}'
        for l in ax.get_yticklabels() + [ax.yaxis.offsetText]:
            assert l.get_visible() == vy, f'Visibility of y axis #{i} is incorrectly {vy}'
        if not vx:
            assert ax.get_xlabel() == ''
        if not vy:
            assert ax.get_ylabel() == ''