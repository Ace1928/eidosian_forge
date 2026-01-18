import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_composite_empty_not_filled(self):
    dist = Distribution([]).opts(filled=False)
    curve = Compositor.collapse_element(dist, backend='matplotlib')
    assert isinstance(curve, Curve)
    assert curve.vdims == [Dimension(('Value_density', 'Density'))]