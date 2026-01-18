import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_arrow_redim_range_aux(self):
    annotations = Arrow(0, 0)
    redimmed = annotations.redim.range(x=(-0.5, 0.5))
    self.assertEqual(redimmed.kdims[0].range, (-0.5, 0.5))