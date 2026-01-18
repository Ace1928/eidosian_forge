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
def test_mpl_bokeh_mpl_via_builders_opts_method_literal_explicit_backend(self):
    img = Image(np.random.rand(10, 10))
    curve = Curve([1, 2, 3])
    overlay = img * curve
    Store.set_current_backend('matplotlib')
    literal = {'Curve': dict(color='orange', backend='matplotlib'), 'Image': dict(cmap='jet', backend='bokeh')}
    styled = overlay.opts(literal)
    mpl_curve_lookup = Store.lookup_options('matplotlib', styled.Curve.I, 'style')
    self.assertEqual(mpl_curve_lookup.kwargs['color'], 'orange')
    mpl_img_lookup = Store.lookup_options('matplotlib', styled.Image.I, 'style')
    self.assertNotEqual(mpl_img_lookup.kwargs['cmap'], 'jet')
    bokeh_curve_lookup = Store.lookup_options('bokeh', styled.Curve.I, 'style')
    self.assertNotEqual(bokeh_curve_lookup.kwargs['color'], 'orange')
    bokeh_img_lookup = Store.lookup_options('bokeh', styled.Image.I, 'style')
    self.assertEqual(bokeh_img_lookup.kwargs['cmap'], 'jet')