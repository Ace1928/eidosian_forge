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
def test_degenerate_polygon():
    point = [0, 0]
    correct_extents = Bbox([point, point]).extents
    assert np.all(Polygon([point]).get_extents().extents == correct_extents)