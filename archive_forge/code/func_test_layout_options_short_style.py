import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
def test_layout_options_short_style(self):
    """
        Short __call__ syntax.
        """
    im = Image(np.random.rand(10, 10))
    layout = (im + im).opts({'Layout': dict({'hspace': 5})})
    self.assertEqual(Store.lookup_options('matplotlib', layout, 'plot').kwargs['hspace'], 5)