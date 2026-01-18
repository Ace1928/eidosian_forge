import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_line_extent_data_coords(self):
    ax = plt.axes()
    ax.plot([0.1, 1.2, 0.8], [0.9, 0.5, 0.8], transform=ax.transData)
    assert_array_equal(ax.dataLim.get_points(), np.array([[0.1, 0.5], [1.2, 0.9]]))