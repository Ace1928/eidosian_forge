import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
def test_intersect_zero_length_segment():
    this_path = Path(np.array([[0, 0], [1, 1]]))
    outline_path = Path(np.array([[1, 0], [0.5, 0.5], [0.5, 0.5], [0, 1]]))
    assert outline_path.intersects_path(this_path)
    assert this_path.intersects_path(outline_path)