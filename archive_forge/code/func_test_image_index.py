import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_index(self):
    image = Image(self.array1)
    self.assertEqual(image[-0.33, -0.25], 3)