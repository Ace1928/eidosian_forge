import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_no_modify():
    triangles = np.array([[3, 2, 0], [3, 1, 0]], dtype=np.int32)
    points = np.array([(0, 0), (0, 1.1), (1, 0), (1, 1)])
    old_triangles = triangles.copy()
    mtri.Triangulation(points[:, 0], points[:, 1], triangles).edges
    assert_array_equal(old_triangles, triangles)