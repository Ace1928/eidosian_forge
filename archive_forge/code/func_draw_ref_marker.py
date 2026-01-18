import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
def draw_ref_marker(y, style, size):
    ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style), s=size)
    if request.getfixturevalue('ext') == 'png':
        ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style), s=size)