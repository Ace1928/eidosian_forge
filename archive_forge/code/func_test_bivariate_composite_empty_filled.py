import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_composite_empty_filled(self):
    dist = Bivariate([]).opts(filled=True)
    contours = Compositor.collapse_element(dist, backend='matplotlib')
    assert isinstance(contours, Polygons)
    assert contours.vdims == [Dimension('Density')]
    assert len(contours) == 0