import numpy as np
from holoviews.element import HeatMap, Image
from .test_plot import TestMPLPlot, mpl38, mpl_renderer
def test_heatmap_invert_yaxis(self):
    hmap = HeatMap([('A', 1, 1), ('B', 2, 2)]).opts(invert_yaxis=True)
    plot = mpl_renderer.get_plot(hmap)
    array = plot.handles['artist'].get_array()
    expected = np.array([1, np.inf, np.inf, 2])
    if mpl38:
        expected = np.array([[1, np.inf], [np.inf, 2]])
    else:
        expected = np.array([1, np.inf, np.inf, 2])
    masked = np.ma.array(expected, mask=np.logical_not(np.isfinite(expected)))
    np.testing.assert_equal(array, masked)