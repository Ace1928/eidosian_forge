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
def test_make_compound_path_empty():
    empty = Path.make_compound_path()
    assert empty.vertices.shape == (0, 2)
    r2 = Path.make_compound_path(empty, empty)
    assert r2.vertices.shape == (0, 2)
    assert r2.codes.shape == (0,)
    r3 = Path.make_compound_path(Path([(0, 0)]), empty)
    assert r3.vertices.shape == (1, 2)
    assert r3.codes.shape == (1,)