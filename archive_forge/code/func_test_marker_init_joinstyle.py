import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
def test_marker_init_joinstyle():
    marker = markers.MarkerStyle('*')
    styled_marker = markers.MarkerStyle('*', joinstyle='round')
    assert styled_marker.get_joinstyle() == 'round'
    assert marker.get_joinstyle() != 'round'