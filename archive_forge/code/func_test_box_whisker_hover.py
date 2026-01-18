import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_hover(self):
    xs, ys = (np.random.randint(0, 5, 100), np.random.randn(100))
    box = BoxWhisker((xs, ys), 'A').sort().opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(box)
    src = plot.handles['vbar_1_source']
    ys = box.aggregate(function=np.median).dimension_values('y')
    hover_tool = plot.handles['hover']
    self.assertEqual(src.data['y'], ys)
    self.assertIn(plot.handles['vbar_1_glyph_renderer'], hover_tool.renderers)
    self.assertIn(plot.handles['vbar_2_glyph_renderer'], hover_tool.renderers)
    self.assertIn(plot.handles['circle_1_glyph_renderer'], hover_tool.renderers)