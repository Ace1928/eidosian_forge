import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@image_comparison(['large_arc.svg'], style='mpl20')
def test_large_arc():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x = 210
    y = -2115
    diameter = 4261
    for ax in [ax1, ax2]:
        a = Arc((x, y), diameter, diameter, lw=2, color='k')
        ax.add_patch(a)
        ax.set_axis_off()
        ax.set_aspect('equal')
    ax1.set_xlim(7, 8)
    ax1.set_ylim(5, 6)
    ax2.set_xlim(-25000, 18000)
    ax2.set_ylim(-20000, 6600)