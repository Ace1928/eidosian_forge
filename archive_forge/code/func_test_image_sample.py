import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_sample(self):
    image = Image(self.array1)
    self.assertEqual(image.sample(y=0.25), Curve(np.array([(-0.333333, 0), (0, 1), (0.333333, 2)]), kdims=['x'], vdims=['z']))