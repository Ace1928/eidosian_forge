import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_composite_transfer_opts(self):
    dist = Distribution(np.array([0, 1, 2])).opts(color='red')
    area = Compositor.collapse_element(dist, backend='matplotlib')
    opts = Store.lookup_options('matplotlib', area, 'style').kwargs
    assert opts.get('color', None) == 'red'