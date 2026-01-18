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
@pytest.mark.parametrize('path, extents', zip(_test_paths, _test_path_extents))
def test_exact_extents(path, extents):
    assert np.all(path.get_extents().extents == extents)