import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
def matshow(self, Z, **kwargs):
    """
        Plot the values of a 2D matrix or array as color-coded image.

        The matrix will be shown the way it would be printed, with the first
        row at the top.  Row and column numbering is zero-based.

        Parameters
        ----------
        Z : (M, N) array-like
            The matrix to be displayed.

        Returns
        -------
        `~matplotlib.image.AxesImage`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.axes.Axes.imshow` arguments

        See Also
        --------
        imshow : More general function to plot data on a 2D regular raster.

        Notes
        -----
        This is just a convenience function wrapping `.imshow` to set useful
        defaults for displaying a matrix. In particular:

        - Set ``origin='upper'``.
        - Set ``interpolation='nearest'``.
        - Set ``aspect='equal'``.
        - Ticks are placed to the left and above.
        - Ticks are formatted to show integer indices.

        """
    Z = np.asanyarray(Z)
    kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'equal', **kwargs}
    im = self.imshow(Z, **kw)
    self.title.set_y(1.05)
    self.xaxis.tick_top()
    self.xaxis.set_ticks_position('both')
    self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    return im