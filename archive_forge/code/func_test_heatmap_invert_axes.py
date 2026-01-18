import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_invert_axes(self):
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    hm = HeatMap(Image(arr)).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(hm)
    xdim, ydim = hm.kdims
    source = plot.handles['source']
    self.assertEqual(source.data['zvalues'], hm.dimension_values(2, flat=False).T.flatten())
    self.assertEqual(source.data['x'], hm.dimension_values(1))
    self.assertEqual(source.data['y'], hm.dimension_values(0))