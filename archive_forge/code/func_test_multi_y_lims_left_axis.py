from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_y_lims_left_axis(self):
    overlay = (Curve(range(10), vdims=['A']).opts(ylim=(-10, 20)) * Curve(range(10), vdims=['B'])).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, -10)
    self.assertEqual(y_range.end, 20)
    extra_y_ranges = plot.handles['extra_y_ranges']
    self.assertEqual(list(extra_y_ranges.keys()), ['B'])
    self.assertEqual(extra_y_ranges['B'].start, 0)
    self.assertEqual(extra_y_ranges['B'].end, 9)