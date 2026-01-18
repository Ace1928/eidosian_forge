import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_from_points(self):
    points = Points(np.array([[0, 1], [1, 2], [2, 3]]))
    dist = Bivariate(points)
    assert dist.kdims == points.kdims