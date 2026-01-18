from unittest import SkipTest
import numpy as np
from matplotlib.colors import ListedColormap
from holoviews.element import Image, ImageStack, Raster
from holoviews.plotting.mpl.raster import RGBPlot
from .test_plot import TestMPLPlot, mpl_renderer
def test_image_cbar_extend_both(self):
    img = Image(np.array([[0, 1], [2, 3]])).redim(z=dict(range=(1, 2)))
    plot = mpl_renderer.get_plot(img.opts(colorbar=True))
    assert plot.handles['cbar'].extend == 'both'