import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
def test_layout_options_long_style(self):
    """
        The old (longer) syntax in __call__
        """
    im = Image(np.random.rand(10, 10))
    layout = (im + im).opts({'Layout': dict({'hspace': 10})})
    self.assertEqual(Store.lookup_options('matplotlib', layout, 'plot').kwargs['hspace'], 10)