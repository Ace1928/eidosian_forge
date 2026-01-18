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
def test_external_transform_api():

    class ScaledBy:

        def __init__(self, scale_factor):
            self._scale_factor = scale_factor

        def _as_mpl_transform(self, axes):
            return mtransforms.Affine2D().scale(self._scale_factor) + axes.transData
    ax = plt.axes()
    line, = plt.plot(np.arange(10), transform=ScaledBy(10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    assert_allclose(line.get_transform()._a.get_matrix(), mtransforms.Affine2D().scale(10).get_matrix())