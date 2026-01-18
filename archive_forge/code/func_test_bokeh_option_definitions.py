import numpy as np
import pyviz_comms as comms
from bokeh.models import (
from param import concrete_descendents
from holoviews import Curve
from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from .. import option_intersections
def test_bokeh_option_definitions(self):
    self.assertEqual(option_intersections('bokeh'), self.known_clashes)