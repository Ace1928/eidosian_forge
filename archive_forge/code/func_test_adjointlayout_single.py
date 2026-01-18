import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_adjointlayout_single(self):
    layout = AdjointLayout([self.view1])
    self.assertEqual(layout.main, self.view1)