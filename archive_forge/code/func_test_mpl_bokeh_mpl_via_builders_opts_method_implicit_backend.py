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
def test_mpl_bokeh_mpl_via_builders_opts_method_implicit_backend(self):
    img = Image(np.random.rand(10, 10))
    Store.set_current_backend('matplotlib')
    mpl_opts = opts.Image(cmap='Blues')
    bokeh_opts = opts.Image(cmap='Purple', backend='bokeh')
    self.assertEqual('backend' not in mpl_opts.kwargs, True)
    self.assertEqual(bokeh_opts.kwargs['backend'], 'bokeh')
    img.opts(mpl_opts, bokeh_opts)
    mpl_lookup = Store.lookup_options('matplotlib', img, 'style').options
    self.assertEqual(mpl_lookup['cmap'], 'Blues')
    bokeh_lookup = Store.lookup_options('bokeh', img, 'style').options
    self.assertEqual(bokeh_lookup['cmap'], 'Purple')
    self.assert_output_options_group_empty(img)