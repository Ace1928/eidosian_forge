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
def barh(self, y, width, height=0.8, left=None, *, align='center', data=None, **kwargs):
    """
        Make a horizontal bar plot.

        The bars are positioned at *y* with the given *align*\\ment. Their
        dimensions are given by *width* and *height*. The horizontal baseline
        is *left* (default 0).

        Many parameters can take either a single value applying to all bars
        or a sequence of values, one for each bar.

        Parameters
        ----------
        y : float or array-like
            The y coordinates of the bars. See also *align* for the
            alignment of the bars to the coordinates.

        width : float or array-like
            The width(s) of the bars.

            Note that if *left* has units (e.g. datetime), *width* should be in
            units that are a difference from the value of *left* (e.g. timedelta).

        height : float or array-like, default: 0.8
            The heights of the bars.

            Note that if *y* has units (e.g. datetime), then *height* should be in
            units that are a difference (e.g. timedelta) around the *y* values.

        left : float or array-like, default: 0
            The x coordinates of the left side(s) of the bars.

            Note that if *left* has units, then the x-axis will get a Locator and
            Formatter appropriate for the units (e.g. dates, or categorical).

        align : {'center', 'edge'}, default: 'center'
            Alignment of the base to the *y* coordinates*:

            - 'center': Center the bars on the *y* positions.
            - 'edge': Align the bottom edges of the bars with the *y*
              positions.

            To align the bars on the top edge pass a negative *height* and
            ``align='edge'``.

        Returns
        -------
        `.BarContainer`
            Container with all the bars and optionally errorbars.

        Other Parameters
        ----------------
        color : color or list of color, optional
            The colors of the bar faces.

        edgecolor : color or list of color, optional
            The colors of the bar edges.

        linewidth : float or array-like, optional
            Width of the bar edge(s). If 0, don't draw edges.

        tick_label : str or list of str, optional
            The tick labels of the bars.
            Default: None (Use default numeric labels.)

        label : str or list of str, optional
            A single label is attached to the resulting `.BarContainer` as a
            label for the whole dataset.
            If a list is provided, it must be the same length as *y* and
            labels the individual bars. Repeated labels are not de-duplicated
            and will cause repeated label entries, so this is best used when
            bars also differ in style (e.g., by passing a list to *color*.)

        xerr, yerr : float or array-like of shape(N,) or shape(2, N), optional
            If not *None*, add horizontal / vertical errorbars to the bar tips.
            The values are +/- sizes relative to the data:

            - scalar: symmetric +/- values for all bars
            - shape(N,): symmetric +/- values for each bar
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar. (default)

            See :doc:`/gallery/statistics/errorbar_features` for an example on
            the usage of *xerr* and *yerr*.

        ecolor : color or list of color, default: 'black'
            The line color of the errorbars.

        capsize : float, default: :rc:`errorbar.capsize`
           The length of the error bar caps in points.

        error_kw : dict, optional
            Dictionary of keyword arguments to be passed to the
            `~.Axes.errorbar` method. Values of *ecolor* or *capsize* defined
            here take precedence over the independent keyword arguments.

        log : bool, default: False
            If ``True``, set the x-axis to be log scale.

        data : indexable object, optional
            If given, all parameters also accept a string ``s``, which is
            interpreted as ``data[s]`` (unless this raises an exception).

        **kwargs : `.Rectangle` properties

        %(Rectangle:kwdoc)s

        See Also
        --------
        bar : Plot a vertical bar plot.

        Notes
        -----
        Stacked bars can be achieved by passing individual *left* values per
        bar. See
        :doc:`/gallery/lines_bars_and_markers/horizontal_barchart_distribution`.
        """
    kwargs.setdefault('orientation', 'horizontal')
    patches = self.bar(x=left, height=height, width=width, bottom=y, align=align, data=data, **kwargs)
    return patches