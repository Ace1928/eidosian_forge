import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_ndlayout_init(self):
    grid = NdLayout([(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)])
    self.assertEqual(grid.shape, (1, 4))