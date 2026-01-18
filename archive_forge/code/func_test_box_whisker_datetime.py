import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_datetime(self):
    times = np.arange(dt.datetime(2017, 1, 1), dt.datetime(2017, 2, 1), dt.timedelta(days=1))
    box = BoxWhisker((times, np.random.rand(len(times))), kdims=['Date'])
    plot = bokeh_renderer.get_plot(box)
    formatted = [box.kdims[0].pprint_value(t) for t in times]
    self.assertTrue(all((cds.data['index'][0] in formatted for cds in plot.state.select(ColumnDataSource) if len(cds.data.get('index', [])))))