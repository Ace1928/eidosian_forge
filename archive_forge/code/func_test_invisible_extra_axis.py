from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_invisible_extra_axis(self):
    overlay = (Curve(range(10), vdims=['A']) * Curve(range(10), vdims=['B']).opts(yaxis=None)).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    assert len(plot.state.yaxis) == 2
    assert plot.state.yaxis[0].visible
    assert not plot.state.yaxis[1].visible