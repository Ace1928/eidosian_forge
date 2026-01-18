from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_inverted_log_ylim_right_axis(self):
    overlay = (Curve(range(10), vdims=['A']) * Curve(range(10), vdims=['B']).opts(invert_yaxis=True, logy=True, ylim=(2, 20))).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    y_range = plot.handles['y_range']
    self.assertEqual(y_range.start, 0)
    self.assertEqual(y_range.end, 9)
    extra_y_ranges = plot.handles['extra_y_ranges']
    self.assertEqual(list(extra_y_ranges.keys()), ['B'])
    print(extra_y_ranges['B'].start, extra_y_ranges['B'].end)
    self.assertEqual(extra_y_ranges['B'].start, 20)
    self.assertEqual(extra_y_ranges['B'].end, 2)
    self.assertTrue(isinstance(plot.handles['extra_y_scales']['B'], LogScale))