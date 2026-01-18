import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_delaunay_points_in_line():
    x = np.linspace(0.0, 10.0, 11)
    y = np.linspace(0.0, 10.0, 11)
    with pytest.raises(RuntimeError):
        mtri.Triangulation(x, y)
    x = np.append(x, 2.0)
    y = np.append(y, 8.0)
    mtri.Triangulation(x, y)