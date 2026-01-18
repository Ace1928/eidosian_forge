from holoviews.element import Div
from .test_plot import TestBokehPlot, bokeh_renderer
def test_div_plot(self):
    html = '<h1>Test</h1>'
    div = Div(html)
    plot = bokeh_renderer.get_plot(div)
    bkdiv = plot.handles['plot']
    self.assertEqual(bkdiv.text, '&lt;h1&gt;Test&lt;/h1&gt;')