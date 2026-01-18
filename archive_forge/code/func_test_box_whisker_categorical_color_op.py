import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_categorical_color_op(self):
    a = np.repeat(np.arange(5), 5)
    b = np.repeat(['A', 'B', 'C', 'D', 'E'], 5)
    box = BoxWhisker((a, b, np.arange(25)), ['a', 'b'], 'd').opts(box_color='b')
    plot = bokeh_renderer.get_plot(box)
    source = plot.handles['vbar_1_source']
    glyph = plot.handles['vbar_1_glyph']
    cmapper = plot.handles['box_color_color_mapper']
    self.assertEqual(source.data['box_color'], b[::5])
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C', 'D', 'E'])
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'box_color', 'transform': cmapper})