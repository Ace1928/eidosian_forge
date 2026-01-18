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
def test_line_extents_non_affine(self):
    ax = plt.axes()
    offset = mtransforms.Affine2D().translate(10, 10)
    na_offset = NonAffineForTest(mtransforms.Affine2D().translate(10, 10))
    plt.plot(np.arange(10), transform=offset + na_offset + ax.transData)
    expected_data_lim = np.array([[0.0, 0.0], [9.0, 9.0]]) + 20
    assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)