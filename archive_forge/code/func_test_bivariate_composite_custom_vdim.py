import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_composite_custom_vdim(self):
    dist = Bivariate(np.random.rand(10, 2), vdims=['Test'])
    contours = Compositor.collapse_element(dist, backend='matplotlib')
    assert isinstance(contours, Contours)
    assert contours.vdims == [Dimension('Test')]