import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_rtol_config(self):
    vals = np.random.rand(20, 20)
    xs = np.linspace(0, 10, 20)
    ys = np.linspace(0, 10, 20)
    ys[-1] += 0.001
    image_rtol = hv.config.image_rtol
    hv.config.image_rtol = 0.01
    Image({'vals': vals, 'xs': xs, 'ys': ys}, ['xs', 'ys'], 'vals')
    hv.config.image_rtol = image_rtol