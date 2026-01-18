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
def step(self, x, y, *args, where='pre', data=None, **kwargs):
    """
        Make a step plot.

        Call signatures::

            step(x, y, [fmt], *, data=None, where='pre', **kwargs)
            step(x, y, [fmt], x2, y2, [fmt2], ..., *, where='pre', **kwargs)

        This is just a thin wrapper around `.plot` which changes some
        formatting options. Most of the concepts and parameters of plot can be
        used here as well.

        .. note::

            This method uses a standard plot with a step drawstyle: The *x*
            values are the reference positions and steps extend left/right/both
            directions depending on *where*.

            For the common case where you know the values and edges of the
            steps, use `~.Axes.stairs` instead.

        Parameters
        ----------
        x : array-like
            1D sequence of x positions. It is assumed, but not checked, that
            it is uniformly increasing.

        y : array-like
            1D sequence of y levels.

        fmt : str, optional
            A format string, e.g. 'g' for a green line. See `.plot` for a more
            detailed description.

            Note: While full format strings are accepted, it is recommended to
            only specify the color. Line styles are currently ignored (use
            the keyword argument *linestyle* instead). Markers are accepted
            and plotted on the given positions, however, this is a rarely
            needed feature for step plots.

        where : {'pre', 'post', 'mid'}, default: 'pre'
            Define where the steps should be placed:

            - 'pre': The y value is continued constantly to the left from
              every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
              value ``y[i]``.
            - 'post': The y value is continued constantly to the right from
              every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
              value ``y[i]``.
            - 'mid': Steps occur half-way between the *x* positions.

        data : indexable object, optional
            An object with labelled data. If given, provide the label names to
            plot in *x* and *y*.

        **kwargs
            Additional parameters are the same as those for `.plot`.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.
        """
    _api.check_in_list(('pre', 'post', 'mid'), where=where)
    kwargs['drawstyle'] = 'steps-' + where
    return self.plot(x, y, *args, data=data, **kwargs)