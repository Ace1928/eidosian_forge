import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_rtol_failure(self):
    vals = np.random.rand(20, 20)
    xs = np.linspace(0, 10, 20)
    ys = np.linspace(0, 10, 20)
    ys[-1] += 0.1
    Image({'vals': vals, 'xs': xs, 'ys': ys}, ['xs', 'ys'], 'vals')
    substr = 'set a higher tolerance on hv.config.image_rtol or the rtol parameter in the Image constructor.'
    self.log_handler.assertEndsWith('WARNING', substr)