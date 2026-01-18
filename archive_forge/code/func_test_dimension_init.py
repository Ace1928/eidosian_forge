import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_init(self):
    Dimension('Test dimension')
    Dimension('Test dimension', cyclic=True)
    Dimension('Test dimension', cyclic=True, type=int)
    Dimension('Test dimension', cyclic=True, type=int, unit='Twilight zones')