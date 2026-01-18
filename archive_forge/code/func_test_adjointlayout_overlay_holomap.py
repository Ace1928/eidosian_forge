import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_overlay_holomap(self):
    layout = self.hmap * (self.view1 << self.view3)
    self.assertEqual(layout.main, self.hmap * self.view1)
    self.assertEqual(layout.right, self.view3)