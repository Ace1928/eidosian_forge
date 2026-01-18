import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_composite(self):
    dist = Distribution(np.array([0, 1, 2]))
    area = Compositor.collapse_element(dist, backend='matplotlib')
    assert isinstance(area, Area)
    assert area.vdims == [Dimension(('Value_density', 'Density'))]