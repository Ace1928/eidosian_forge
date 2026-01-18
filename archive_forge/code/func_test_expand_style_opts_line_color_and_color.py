import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_expand_style_opts_line_color_and_color(self):
    style = {'fill_color': 'red', 'color': 'blue'}
    opts = ['color', 'line_color', 'fill_color']
    data, mapping = expand_batched_style(style, opts, {}, nvals=3)
    self.assertEqual(data['line_color'], ['blue', 'blue', 'blue'])
    self.assertEqual(data['fill_color'], ['red', 'red', 'red'])
    self.assertEqual(mapping, {'line_color': {'field': 'line_color'}, 'fill_color': {'field': 'fill_color'}})