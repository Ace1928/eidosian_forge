import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_expand_style_opts_simple(self):
    style = {'line_width': 3}
    opts = ['line_width']
    data, mapping = expand_batched_style(style, opts, {}, nvals=3)
    self.assertEqual(data['line_width'], [3, 3, 3])
    self.assertEqual(mapping, {'line_width': {'field': 'line_width'}})