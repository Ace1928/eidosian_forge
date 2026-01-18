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
def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
    """
        Add a vertical span (rectangle) across the Axes.

        The rectangle spans from *xmin* to *xmax* horizontally, and, by
        default, the whole y-axis vertically.  The y-span can be set using
        *ymin* (default: 0) and *ymax* (default: 1) which are in axis units;
        e.g. ``ymin = 0.5`` always refers to the middle of the y-axis
        regardless of the limits set by `~.Axes.set_ylim`.

        Parameters
        ----------
        xmin : float
            Lower x-coordinate of the span, in data units.
        xmax : float
            Upper x-coordinate of the span, in data units.
        ymin : float, default: 0
            Lower y-coordinate of the span, in y-axis units (0-1).
        ymax : float, default: 1
            Upper y-coordinate of the span, in y-axis units (0-1).

        Returns
        -------
        `~matplotlib.patches.Polygon`
            Vertical span (rectangle) from (xmin, ymin) to (xmax, ymax).

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties

        %(Polygon:kwdoc)s

        See Also
        --------
        axhspan : Add a horizontal span across the Axes.

        Examples
        --------
        Draw a vertical, green, translucent rectangle from x = 1.25 to
        x = 1.55 that spans the yrange of the Axes.

        >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

        """
    self._check_no_units([ymin, ymax], ['ymin', 'ymax'])
    (xmin, xmax), = self._process_unit_info([('x', [xmin, xmax])], kwargs)
    verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    p = mpatches.Polygon(verts, **kwargs)
    p.set_transform(self.get_xaxis_transform(which='grid'))
    p.get_path()._interpolation_steps = 100
    self.add_patch(p)
    self._request_autoscale_view('x')
    return p