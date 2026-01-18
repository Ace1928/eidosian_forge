import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_cyclic_property_true(self):
    cycle1 = Cycle(values=['a', 'b', 'c'])
    opts = Options('test', one=cycle1, two='two')
    self.assertEqual(opts.cyclic, True)