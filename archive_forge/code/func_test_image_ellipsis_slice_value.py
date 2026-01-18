import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_image_ellipsis_slice_value(self):
    data = np.random.rand(10, 10)
    sliced = hv.Image(data)[..., 'z']
    self.assertEqual(sliced.data, data)