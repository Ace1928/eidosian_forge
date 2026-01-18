from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_multi_log_right_axis_warn(self):
    overlay = (Curve(range(10), vdims=['A']) * Curve(range(10), vdims=['B']).opts(logy=True)).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    self.assertEqual(len(plot.state.yaxis), 2)
    self.assertTrue(isinstance(plot.state.yaxis[0], LinearAxis))
    self.assertTrue(isinstance(plot.state.yaxis[1], LogAxis))
    extra_y_ranges = plot.handles['extra_y_scales']
    self.assertTrue(isinstance(extra_y_ranges['B'], LogScale))
    substr = 'Logarithmic axis range encountered value less than or equal to zero, please supply explicit lower bound to override default of 0.010.'
    self.log_handler.assertEndsWith('WARNING', substr)