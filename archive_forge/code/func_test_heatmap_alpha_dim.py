import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_alpha_dim(self):
    data = {'row': [1, 2, 1, 2], 'col': [1, 2, 2, 1], 'alpha': [0, 0, 0, 1], 'val': [0.5, 0.6, 0.2, 0.1]}
    hm = HeatMap(data, kdims=['col', 'row'], vdims=['val', 'alpha']).opts(alpha='alpha')
    plot = bokeh_renderer.get_plot(hm)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['row'], np.array([1, 2, 1, 2]))
    self.assertEqual(cds.data['col'], np.array([1, 1, 2, 2]))
    self.assertEqual(cds.data['alpha'], np.array([0, 1, 0, 0]))
    self.assertEqual(cds.data['zvalues'], np.array([0.5, 0.1, 0.2, 0.6]))