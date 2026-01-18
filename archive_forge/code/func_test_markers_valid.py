import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@pytest.mark.parametrize('marker', ['o', 'x', '', 'None', '$\\frac{1}{2}$', '$♫$', 1, markers.TICKLEFT, [[-1, 0], [1, 0]], np.array([[-1, 0], [1, 0]]), Path([[0, 0], [1, 0]], [Path.MOVETO, Path.LINETO]), (5, 0), (7, 1), (5, 2), (5, 0, 10), (7, 1, 10), (5, 2, 10), markers.MarkerStyle('o')])
def test_markers_valid(marker):
    markers.MarkerStyle(marker)