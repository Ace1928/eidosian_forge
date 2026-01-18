import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_delaunay_duplicate_points():
    npoints = 10
    duplicate = 7
    duplicate_of = 3
    np.random.seed(23)
    x = np.random.random(npoints)
    y = np.random.random(npoints)
    x[duplicate] = x[duplicate_of]
    y[duplicate] = y[duplicate_of]
    triang = mtri.Triangulation(x, y)
    assert_array_equal(np.unique(triang.triangles), np.delete(np.arange(npoints), duplicate))