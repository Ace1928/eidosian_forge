import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
def test_marker_rotated_invalid():
    marker = markers.MarkerStyle('o')
    with pytest.raises(ValueError):
        new_marker = marker.rotated()
    with pytest.raises(ValueError):
        new_marker = marker.rotated(deg=10, rad=10)