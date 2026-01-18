import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_default_antialiased():
    patch = Patch()
    patch.set_antialiased(not rcParams['patch.antialiased'])
    assert patch.get_antialiased() == (not rcParams['patch.antialiased'])
    patch.set_antialiased(None)
    assert patch.get_antialiased() == rcParams['patch.antialiased']