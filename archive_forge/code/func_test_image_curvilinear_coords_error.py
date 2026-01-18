import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_curvilinear_coords_error(self):
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X ** 2 + Y ** 2) * np.cos(X)
    with self.assertRaises(ValueError):
        Image((X, Y, Z))