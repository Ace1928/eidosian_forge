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
@image_comparison(['autoscale_arc'], extensions=['png', 'svg'], style='mpl20', remove_text=True)
def test_autoscale_arc():
    fig, axs = plt.subplots(1, 3, figsize=(4, 1))
    arc_lists = ([Arc((0, 0), 1, 1, theta1=0, theta2=90)], [Arc((0.5, 0.5), 1.5, 0.5, theta1=10, theta2=20)], [Arc((0.5, 0.5), 1.5, 0.5, theta1=10, theta2=20), Arc((0.5, 0.5), 2.5, 0.5, theta1=110, theta2=120), Arc((0.5, 0.5), 3.5, 0.5, theta1=210, theta2=220), Arc((0.5, 0.5), 4.5, 0.5, theta1=310, theta2=320)])
    for ax, arcs in zip(axs, arc_lists):
        for arc in arcs:
            ax.add_patch(arc)
        ax.autoscale()