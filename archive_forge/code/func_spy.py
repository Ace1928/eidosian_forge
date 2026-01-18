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
@_docstring.dedent_interpd
def spy(self, Z, precision=0, marker=None, markersize=None, aspect='equal', origin='upper', **kwargs):
    """
        Plot the sparsity pattern of a 2D array.

        This visualizes the non-zero values of the array.

        Two plotting styles are available: image and marker. Both
        are available for full arrays, but only the marker style
        works for `scipy.sparse.spmatrix` instances.

        **Image style**

        If *marker* and *markersize* are *None*, `~.Axes.imshow` is used. Any
        extra remaining keyword arguments are passed to this method.

        **Marker style**

        If *Z* is a `scipy.sparse.spmatrix` or *marker* or *markersize* are
        *None*, a `.Line2D` object will be returned with the value of marker
        determining the marker type, and any remaining keyword arguments
        passed to `~.Axes.plot`.

        Parameters
        ----------
        Z : (M, N) array-like
            The array to be plotted.

        precision : float or 'present', default: 0
            If *precision* is 0, any non-zero value will be plotted. Otherwise,
            values of :math:`|Z| > precision` will be plotted.

            For `scipy.sparse.spmatrix` instances, you can also
            pass 'present'. In this case any value present in the array
            will be plotted, even if it is identically zero.

        aspect : {'equal', 'auto', None} or float, default: 'equal'
            The aspect ratio of the Axes.  This parameter is particularly
            relevant for images since it determines whether data pixels are
            square.

            This parameter is a shortcut for explicitly calling
            `.Axes.set_aspect`. See there for further details.

            - 'equal': Ensures an aspect ratio of 1. Pixels will be square.
            - 'auto': The Axes is kept fixed and the aspect is adjusted so
              that the data fit in the Axes. In general, this will result in
              non-square pixels.
            - *None*: Use :rc:`image.aspect`.

        origin : {'upper', 'lower'}, default: :rc:`image.origin`
            Place the [0, 0] index of the array in the upper left or lower left
            corner of the Axes. The convention 'upper' is typically used for
            matrices and images.

        Returns
        -------
        `~matplotlib.image.AxesImage` or `.Line2D`
            The return type depends on the plotting style (see above).

        Other Parameters
        ----------------
        **kwargs
            The supported additional parameters depend on the plotting style.

            For the image style, you can pass the following additional
            parameters of `~.Axes.imshow`:

            - *cmap*
            - *alpha*
            - *url*
            - any `.Artist` properties (passed on to the `.AxesImage`)

            For the marker style, you can pass any `.Line2D` property except
            for *linestyle*:

            %(Line2D:kwdoc)s
        """
    if marker is None and markersize is None and hasattr(Z, 'tocoo'):
        marker = 's'
    _api.check_in_list(['upper', 'lower'], origin=origin)
    if marker is None and markersize is None:
        Z = np.asarray(Z)
        mask = np.abs(Z) > precision
        if 'cmap' not in kwargs:
            kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'], name='binary')
        if 'interpolation' in kwargs:
            raise _api.kwarg_error('spy', 'interpolation')
        if 'norm' not in kwargs:
            kwargs['norm'] = mcolors.NoNorm()
        ret = self.imshow(mask, interpolation='nearest', aspect=aspect, origin=origin, **kwargs)
    else:
        if hasattr(Z, 'tocoo'):
            c = Z.tocoo()
            if precision == 'present':
                y = c.row
                x = c.col
            else:
                nonzero = np.abs(c.data) > precision
                y = c.row[nonzero]
                x = c.col[nonzero]
        else:
            Z = np.asarray(Z)
            nonzero = np.abs(Z) > precision
            y, x = np.nonzero(nonzero)
        if marker is None:
            marker = 's'
        if markersize is None:
            markersize = 10
        if 'linestyle' in kwargs:
            raise _api.kwarg_error('spy', 'linestyle')
        ret = mlines.Line2D(x, y, linestyle='None', marker=marker, markersize=markersize, **kwargs)
        self.add_line(ret)
        nr, nc = Z.shape
        self.set_xlim(-0.5, nc - 0.5)
        if origin == 'upper':
            self.set_ylim(nr - 0.5, -0.5)
        else:
            self.set_ylim(-0.5, nr - 0.5)
        self.set_aspect(aspect)
    self.title.set_y(1.05)
    if origin == 'upper':
        self.xaxis.tick_top()
    else:
        self.xaxis.tick_bottom()
    self.xaxis.set_ticks_position('both')
    self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
    return ret