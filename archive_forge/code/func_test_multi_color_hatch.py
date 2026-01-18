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
@image_comparison(['multi_color_hatch'], remove_text=True, style='default')
def test_multi_color_hatch():
    fig, ax = plt.subplots()
    rects = ax.bar(range(5), range(1, 6))
    for i, rect in enumerate(rects):
        rect.set_facecolor('none')
        rect.set_edgecolor(f'C{i}')
        rect.set_hatch('/')
    ax.autoscale_view()
    ax.autoscale(False)
    for i in range(5):
        with mpl.style.context({'hatch.color': f'C{i}'}):
            r = Rectangle((i - 0.8 / 2, 5), 0.8, 1, hatch='//', fc='none')
        ax.add_patch(r)