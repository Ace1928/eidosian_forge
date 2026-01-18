import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_add_operator(self):
    layout1 = self.view3 << self.view2
    layout2 = self.view2 << self.view1
    self.assertEqual(type(layout1 + layout2), Layout)