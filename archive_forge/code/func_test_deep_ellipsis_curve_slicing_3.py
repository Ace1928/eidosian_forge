import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_deep_ellipsis_curve_slicing_3(self):
    hmap = hv.HoloMap({i: hv.Curve([(j, 2 * j) for j in range(10)]) for i in range(10)})
    sliced = hmap[..., 2:5]
    self.assertEqual(sliced.last.range('y'), (2, 4))