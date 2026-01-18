import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_array_range_vdims(self):
    dist = Distribution(np.array([0, 1, 2]))
    dmin, dmax = dist.range(1)
    assert not np.isfinite(dmin)
    assert not np.isfinite(dmax)