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
def bar_label(self, container, labels=None, *, fmt='%g', label_type='edge', padding=0, **kwargs):
    """
        Label a bar plot.

        Adds labels to bars in the given `.BarContainer`.
        You may need to adjust the axis limits to fit the labels.

        Parameters
        ----------
        container : `.BarContainer`
            Container with all the bars and optionally errorbars, likely
            returned from `.bar` or `.barh`.

        labels : array-like, optional
            A list of label texts, that should be displayed. If not given, the
            label texts will be the data values formatted with *fmt*.

        fmt : str or callable, default: '%g'
            An unnamed %-style or {}-style format string for the label or a
            function to call with the value as the first argument.
            When *fmt* is a string and can be interpreted in both formats,
            %-style takes precedence over {}-style.

            .. versionadded:: 3.7
               Support for {}-style format string and callables.

        label_type : {'edge', 'center'}, default: 'edge'
            The label type. Possible values:

            - 'edge': label placed at the end-point of the bar segment, and the
              value displayed will be the position of that end-point.
            - 'center': label placed in the center of the bar segment, and the
              value displayed will be the length of that segment.
              (useful for stacked bars, i.e.,
              :doc:`/gallery/lines_bars_and_markers/bar_label_demo`)

        padding : float, default: 0
            Distance of label from the end of the bar, in points.

        **kwargs
            Any remaining keyword arguments are passed through to
            `.Axes.annotate`. The alignment parameters (
            *horizontalalignment* / *ha*, *verticalalignment* / *va*) are
            not supported because the labels are automatically aligned to
            the bars.

        Returns
        -------
        list of `.Annotation`
            A list of `.Annotation` instances for the labels.
        """
    for key in ['horizontalalignment', 'ha', 'verticalalignment', 'va']:
        if key in kwargs:
            raise ValueError(f'Passing {key!r} to bar_label() is not supported.')
    a, b = self.yaxis.get_view_interval()
    y_inverted = a > b
    c, d = self.xaxis.get_view_interval()
    x_inverted = c > d

    def sign(x):
        return 1 if x >= 0 else -1
    _api.check_in_list(['edge', 'center'], label_type=label_type)
    bars = container.patches
    errorbar = container.errorbar
    datavalues = container.datavalues
    orientation = container.orientation
    if errorbar:
        lines = errorbar.lines
        barlinecols = lines[2]
        barlinecol = barlinecols[0]
        errs = barlinecol.get_segments()
    else:
        errs = []
    if labels is None:
        labels = []
    annotations = []
    for bar, err, dat, lbl in itertools.zip_longest(bars, errs, datavalues, labels):
        (x0, y0), (x1, y1) = bar.get_bbox().get_points()
        xc, yc = ((x0 + x1) / 2, (y0 + y1) / 2)
        if orientation == 'vertical':
            extrema = max(y0, y1) if dat >= 0 else min(y0, y1)
            length = abs(y0 - y1)
        else:
            extrema = max(x0, x1) if dat >= 0 else min(x0, x1)
            length = abs(x0 - x1)
        if err is None or np.size(err) == 0:
            endpt = extrema
        elif orientation == 'vertical':
            endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
        else:
            endpt = err[:, 0].max() if dat >= 0 else err[:, 0].min()
        if label_type == 'center':
            value = sign(dat) * length
        else:
            value = extrema
        if label_type == 'center':
            xy = (0.5, 0.5)
            kwargs['xycoords'] = lambda r, b=bar: mtransforms.Bbox.intersection(b.get_window_extent(r), b.get_clip_box()) or mtransforms.Bbox.null()
        elif orientation == 'vertical':
            xy = (xc, endpt)
        else:
            xy = (endpt, yc)
        if orientation == 'vertical':
            y_direction = -1 if y_inverted else 1
            xytext = (0, y_direction * sign(dat) * padding)
        else:
            x_direction = -1 if x_inverted else 1
            xytext = (x_direction * sign(dat) * padding, 0)
        if label_type == 'center':
            ha, va = ('center', 'center')
        elif orientation == 'vertical':
            ha = 'center'
            if y_inverted:
                va = 'top' if dat > 0 else 'bottom'
            else:
                va = 'top' if dat < 0 else 'bottom'
        else:
            if x_inverted:
                ha = 'right' if dat > 0 else 'left'
            else:
                ha = 'right' if dat < 0 else 'left'
            va = 'center'
        if np.isnan(dat):
            lbl = ''
        if lbl is None:
            if isinstance(fmt, str):
                lbl = cbook._auto_format_str(fmt, value)
            elif callable(fmt):
                lbl = fmt(value)
            else:
                raise TypeError('fmt must be a str or callable')
        annotation = self.annotate(lbl, xy, xytext, textcoords='offset points', ha=ha, va=va, **kwargs)
        annotations.append(annotation)
    return annotations