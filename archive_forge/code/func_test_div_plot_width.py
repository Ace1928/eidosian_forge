from holoviews.element import Div
from .test_plot import TestBokehPlot, bokeh_renderer
def test_div_plot_width(self):
    html = '<h1>Test</h1>'
    div = Div(html).opts(width=342, height=432, backend='bokeh')
    plot = bokeh_renderer.get_plot(div)
    bkdiv = plot.handles['plot']
    self.assertEqual(bkdiv.width, 342)
    self.assertEqual(bkdiv.height, 432)