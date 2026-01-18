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
@pytest.mark.parametrize('kwarg', ('edgecolor', 'facecolor'))
def test_color_override_warning(kwarg):
    with pytest.warns(UserWarning, match="Setting the 'color' property will override the edgecolor or facecolor properties."):
        Patch(color='black', **{kwarg: 'black'})