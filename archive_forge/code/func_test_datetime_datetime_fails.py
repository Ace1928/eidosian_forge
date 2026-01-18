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
def test_datetime_datetime_fails():
    from datetime import datetime
    start = datetime(2017, 1, 1, 0, 0, 0)
    dt_delta = datetime(1970, 1, 5)
    with pytest.raises(TypeError):
        mpatches.Rectangle((start, 0), dt_delta, 1)
    with pytest.raises(TypeError):
        mpatches.Rectangle((0, start), 1, dt_delta)