import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_layout_overlay_ncols_preserved(self):
    assert ((self.view1 + self.view2).cols(1) * self.view3)._max_cols == 1