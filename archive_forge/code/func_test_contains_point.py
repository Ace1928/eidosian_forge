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
def test_contains_point():
    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0)
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    path = ell.get_path()
    transform = ell.get_transform()
    radius = ell._process_radius(None)
    expected = np.array([path.contains_point(point, transform, radius) for point in points])
    result = np.array([ell.contains_point(point) for point in points])
    assert np.all(result == expected)