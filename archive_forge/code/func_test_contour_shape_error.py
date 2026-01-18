import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
@pytest.mark.parametrize('args, message', [((np.arange(9), np.arange(9), np.empty((9, 10))), 'Length of x (9) must match number of columns in z (10)'), ((np.arange(10), np.arange(10), np.empty((9, 10))), 'Length of y (10) must match number of rows in z (9)'), ((np.empty((10, 10)), np.arange(10), np.empty((9, 10))), 'Number of dimensions of x (2) and y (1) do not match'), ((np.arange(10), np.empty((10, 10)), np.empty((9, 10))), 'Number of dimensions of x (1) and y (2) do not match'), ((np.empty((9, 9)), np.empty((9, 10)), np.empty((9, 10))), 'Shapes of x (9, 9) and z (9, 10) do not match'), ((np.empty((9, 10)), np.empty((9, 9)), np.empty((9, 10))), 'Shapes of y (9, 9) and z (9, 10) do not match'), ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((9, 10))), 'Inputs x and y must be 1D or 2D, not 3D'), ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((3, 3, 3))), 'Input z must be 2D, not 3D'), (([[0]],), 'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)'), (([0], [0], [[0]]), 'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)')])
def test_contour_shape_error(args, message):
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match=re.escape(message)):
        ax.contour(*args)