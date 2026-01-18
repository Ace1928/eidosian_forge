import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_array_range_vdims(self):
    dist = Bivariate(np.array([[0, 1, 2], [0, 1, 3]]))
    dmin, dmax = dist.range(2)
    assert not np.isfinite(dmin)
    assert not np.isfinite(dmax)