import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_overlay_main_and_right_v1(self):
    layout = (self.view1 << self.view2) * (self.view1 << self.view3)
    self.assertEqual(layout.main, self.view1 * self.view1)
    self.assertEqual(layout.right, self.view2 * self.view3)