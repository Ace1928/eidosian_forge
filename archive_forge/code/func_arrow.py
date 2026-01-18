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
def arrow(self, x, y, dx, dy, **kwargs):
    """
        Add an arrow to the Axes.

        This draws an arrow from ``(x, y)`` to ``(x+dx, y+dy)``.

        Parameters
        ----------
        %(FancyArrow)s

        Returns
        -------
        `.FancyArrow`
            The created `.FancyArrow` object.

        Notes
        -----
        The resulting arrow is affected by the Axes aspect ratio and limits.
        This may produce an arrow whose head is not square with its stem. To
        create an arrow whose head is square with its stem,
        use :meth:`annotate` for example:

        >>> ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        ...             arrowprops=dict(arrowstyle="->"))

        """
    x = self.convert_xunits(x)
    y = self.convert_yunits(y)
    dx = self.convert_xunits(dx)
    dy = self.convert_yunits(dy)
    a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)
    self.add_patch(a)
    self._request_autoscale_view()
    return a