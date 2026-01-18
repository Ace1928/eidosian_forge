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
def test_mpl_bokeh_mpl_via_option_objects_opts_method(self):
    img = Image(np.random.rand(10, 10))
    mpl_opts = Options('Image', cmap='Blues', backend='matplotlib')
    bokeh_opts = Options('Image', cmap='Purple', backend='bokeh')
    self.assertEqual(mpl_opts.kwargs['backend'], 'matplotlib')
    self.assertEqual(bokeh_opts.kwargs['backend'], 'bokeh')
    img.opts(mpl_opts, bokeh_opts)
    mpl_lookup = Store.lookup_options('matplotlib', img, 'style').options
    self.assertEqual(mpl_lookup['cmap'], 'Blues')
    bokeh_lookup = Store.lookup_options('bokeh', img, 'style').options
    self.assertEqual(bokeh_lookup['cmap'], 'Purple')
    self.assert_output_options_group_empty(img)