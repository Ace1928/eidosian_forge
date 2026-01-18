import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
def test_holomap_options_empty_no_exception(self):
    HoloMap({0: Image(np.random.rand(10, 10))}).options()