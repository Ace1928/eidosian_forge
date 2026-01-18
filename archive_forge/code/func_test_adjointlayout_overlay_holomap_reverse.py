import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_overlay_holomap_reverse(self):
    layout = (self.view1 << self.view3) * self.hmap
    self.assertEqual(layout.main, self.view1 * self.hmap)
    self.assertEqual(layout.right, self.view3)