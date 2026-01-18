import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_composite_transfer_opts(self):
    dist = Bivariate(np.random.rand(10, 2)).opts(cmap='Blues')
    contours = Compositor.collapse_element(dist, backend='matplotlib')
    opts = Store.lookup_options('matplotlib', contours, 'style').kwargs
    assert opts.get('cmap', None) == 'Blues'