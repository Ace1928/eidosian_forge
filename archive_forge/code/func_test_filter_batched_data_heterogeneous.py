import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_filter_batched_data_heterogeneous(self):
    data = {'line_color': ['red', 'red', 'blue']}
    mapping = {'line_color': {'field': 'line_color'}}
    filter_batched_data(data, mapping)
    self.assertEqual(data, {'line_color': ['red', 'red', 'blue']})
    self.assertEqual(mapping, {'line_color': {'field': 'line_color'}})