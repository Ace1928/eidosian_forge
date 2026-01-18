import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def meshgrid_triangles(n):
    """
    Return (2*(N-1)**2, 3) array of triangles to mesh (N, N)-point np.meshgrid.
    """
    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i + j * n
            b = i + 1 + j * n
            c = i + (j + 1) * n
            d = i + 1 + (j + 1) * n
            tri += [[a, b, d], [a, d, c]]
    return np.array(tri, dtype=np.int32)