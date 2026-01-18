import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_gridspace_overlay_reverse(self):
    items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
    grid = GridSpace(items, 'X')
    hline = HLine(0)
    overlaid_grid = hline * grid
    expected = GridSpace([(k, hline * v) for k, v in items], 'X')
    self.assertEqual(overlaid_grid, expected)