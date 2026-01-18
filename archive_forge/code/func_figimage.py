from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
@_docstring.interpd
def figimage(self, X, xo=0, yo=0, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, origin=None, resize=False, **kwargs):
    """
        Add a non-resampled image to the figure.

        The image is attached to the lower or upper left corner depending on
        *origin*.

        Parameters
        ----------
        X
            The image data. This is an array of one of the following shapes:

            - (M, N): an image with scalar data.  Color-mapping is controlled
              by *cmap*, *norm*, *vmin*, and *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

        xo, yo : int
            The *x*/*y* image offset in pixels.

        alpha : None or float
            The alpha blending value.

        %(cmap_doc)s

            This parameter is ignored if *X* is RGB(A).

        %(norm_doc)s

            This parameter is ignored if *X* is RGB(A).

        %(vmin_vmax_doc)s

            This parameter is ignored if *X* is RGB(A).

        origin : {'upper', 'lower'}, default: :rc:`image.origin`
            Indicates where the [0, 0] index of the array is in the upper left
            or lower left corner of the axes.

        resize : bool
            If *True*, resize the figure to match the given image size.

        Returns
        -------
        `matplotlib.image.FigureImage`

        Other Parameters
        ----------------
        **kwargs
            Additional kwargs are `.Artist` kwargs passed on to `.FigureImage`.

        Notes
        -----
        figimage complements the Axes image (`~matplotlib.axes.Axes.imshow`)
        which will be resampled to fit the current Axes.  If you want
        a resampled image to fill the entire figure, you can define an
        `~matplotlib.axes.Axes` with extent [0, 0, 1, 1].

        Examples
        --------
        ::

            f = plt.figure()
            nx = int(f.get_figwidth() * f.dpi)
            ny = int(f.get_figheight() * f.dpi)
            data = np.random.random((ny, nx))
            f.figimage(data)
            plt.show()
        """
    if resize:
        dpi = self.get_dpi()
        figsize = [x / dpi for x in (X.shape[1], X.shape[0])]
        self.set_size_inches(figsize, forward=True)
    im = mimage.FigureImage(self, cmap=cmap, norm=norm, offsetx=xo, offsety=yo, origin=origin, **kwargs)
    im.stale_callback = _stale_figure_callback
    im.set_array(X)
    im.set_alpha(alpha)
    if norm is None:
        im.set_clim(vmin, vmax)
    self.images.append(im)
    im._remove_method = self.images.remove
    self.stale = True
    return im