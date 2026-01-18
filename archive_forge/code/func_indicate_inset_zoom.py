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
def indicate_inset_zoom(self, inset_ax, **kwargs):
    """
        Add an inset indicator rectangle to the Axes based on the axis
        limits for an *inset_ax* and draw connectors between *inset_ax*
        and the rectangle.

        Warnings
        --------
        This method is experimental as of 3.0, and the API may change.

        Parameters
        ----------
        inset_ax : `.Axes`
            Inset Axes to draw connecting lines to.  Two lines are
            drawn connecting the indicator box to the inset Axes on corners
            chosen so as to not overlap with the indicator box.

        **kwargs
            Other keyword arguments are passed on to `.Axes.indicate_inset`

        Returns
        -------
        rectangle_patch : `.patches.Rectangle`
             Rectangle artist.

        connector_lines : 4-tuple of `.patches.ConnectionPatch`
            Each of four connector lines coming from the rectangle drawn on
            this axis, in the order lower left, upper left, lower right,
            upper right.
            Two are set with visibility to *False*,  but the user can
            set the visibility to *True* if the automatic choice is not deemed
            correct.
        """
    xlim = inset_ax.get_xlim()
    ylim = inset_ax.get_ylim()
    rect = (xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0])
    return self.indicate_inset(rect, inset_ax, **kwargs)