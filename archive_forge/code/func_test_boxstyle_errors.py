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
@pytest.mark.parametrize('fmt, match', (('foo', "Unknown style: 'foo'"), ('Round,foo', "Incorrect style argument: 'Round,foo'")))
def test_boxstyle_errors(fmt, match):
    with pytest.raises(ValueError, match=match):
        BoxStyle(fmt)