from unittest import SkipTest
import numpy as np
from matplotlib.colors import ListedColormap
from holoviews.element import Image, ImageStack, Raster
from holoviews.plotting.mpl.raster import RGBPlot
from .test_plot import TestMPLPlot, mpl_renderer
def test_image_cbar_extend_clim(self):
    img = Image(np.array([[0, 1], [2, 3]])).opts(clim=(np.nan, np.nan), colorbar=True)
    plot = mpl_renderer.get_plot(img)
    assert plot.handles['cbar'].extend == 'neither'