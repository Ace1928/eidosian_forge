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
def test_mpl_bokeh_output_options_group_expandable(self):
    original_allowed_kws = Options._output_allowed_kws[:]
    Options._output_allowed_kws = ['backend', 'file_format_example']
    Store.register({Curve: plotting.mpl.CurvePlot}, 'matplotlib')
    Store.register({Curve: plotting.bokeh.CurvePlot}, 'bokeh')
    curve_bk = Options('Curve', backend='bokeh', color='blue')
    curve_mpl = Options('Curve', backend='matplotlib', color='red', file_format_example='SVG')
    c = Curve([1, 2, 3])
    styled = c.opts(curve_bk, curve_mpl)
    self.assertEqual(Store.lookup_options('matplotlib', styled, 'output').kwargs, {'backend': 'matplotlib', 'file_format_example': 'SVG'})
    self.assertEqual(Store.lookup_options('bokeh', styled, 'output').kwargs, {})
    Options._output_allowed_kws = original_allowed_kws