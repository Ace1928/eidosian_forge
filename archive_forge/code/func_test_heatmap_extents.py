import numpy as np
from holoviews.element import HeatMap, Image
from .test_plot import TestMPLPlot, mpl38, mpl_renderer
def test_heatmap_extents(self):
    hmap = HeatMap([('A', 50, 1), ('B', 2, 2), ('C', 50, 1)])
    plot = mpl_renderer.get_plot(hmap)
    assert plot.get_extents(hmap, {}) == (-0.5, -22, 2.5, 74)