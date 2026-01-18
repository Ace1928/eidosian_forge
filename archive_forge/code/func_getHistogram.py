import warnings
import numpy as np
from .. import functions as fn
from ..colormap import ColorMap
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .HistogramLUTItem import HistogramLUTItem
def getHistogram(self, **kwds):
    """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().
        """
    z = self.data[2]
    z = z[np.isfinite(z)]
    hist = np.histogram(z, **kwds)
    return (hist[1][:-1], hist[0])