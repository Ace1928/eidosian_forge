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
@pytest.mark.parametrize('other_path, inside, inverted_inside', [(Path([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25), (0.25, 0.25)], closed=True), True, False), (Path([(-0.25, -0.25), (-0.25, 1.75), (1.75, 1.75), (1.75, -0.25), (-0.25, -0.25)], closed=True), False, True), (Path([(-0.25, -0.25), (-0.25, 1.75), (0.5, 0.5), (1.75, 1.75), (1.75, -0.25), (-0.25, -0.25)], closed=True), False, False), (Path([(0.25, 0.25), (0.25, 1.25), (1.25, 1.25), (1.25, 0.25), (0.25, 0.25)], closed=True), False, False), (Path([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)], closed=True), False, False), (Path([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)], closed=True), False, False)])
def test_contains_path(other_path, inside, inverted_inside):
    path = Path([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)], closed=True)
    assert path.contains_path(other_path) is inside
    assert other_path.contains_path(path) is inverted_inside