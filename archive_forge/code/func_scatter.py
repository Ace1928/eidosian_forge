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
@_preprocess_data(replace_names=['x', 'y', 's', 'linewidths', 'edgecolors', 'c', 'facecolor', 'facecolors', 'color'], label_namer='y')
@_docstring.interpd
def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, **kwargs):
    """
        A scatter plot of *y* vs. *x* with varying marker size and/or color.

        Parameters
        ----------
        x, y : float or array-like, shape (n, )
            The data positions.

        s : float or array-like, shape (n, ), optional
            The marker size in points**2 (typographic points are 1/72 in.).
            Default is ``rcParams['lines.markersize'] ** 2``.

            The linewidth and edgecolor can visually interact with the marker
            size, and can lead to artifacts if the marker size is smaller than
            the linewidth.

            If the linewidth is greater than 0 and the edgecolor is anything
            but *'none'*, then the effective size of the marker will be
            increased by half the linewidth because the stroke will be centered
            on the edge of the shape.

            To eliminate the marker edge either set *linewidth=0* or
            *edgecolor='none'*.

        c : array-like or list of colors or color, optional
            The marker colors. Possible values:

            - A scalar or sequence of n numbers to be mapped to colors using
              *cmap* and *norm*.
            - A 2D array in which the rows are RGB or RGBA.
            - A sequence of colors of length n.
            - A single color format string.

            Note that *c* should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values to be
            colormapped. If you want to specify the same RGB or RGBA value for
            all points, use a 2D array with a single row.  Otherwise,
            value-matching will have precedence in case of a size matching with
            *x* and *y*.

            If you wish to specify a single color for all points
            prefer the *color* keyword argument.

            Defaults to `None`. In that case the marker color is determined
            by the value of *color*, *facecolor* or *facecolors*. In case
            those are not specified or `None`, the marker color is determined
            by the next color of the ``Axes``' current "shape and fill" color
            cycle. This cycle defaults to :rc:`axes.prop_cycle`.

        marker : `~.markers.MarkerStyle`, default: :rc:`scatter.marker`
            The marker style. *marker* can be either an instance of the class
            or the text shorthand for a particular marker.
            See :mod:`matplotlib.markers` for more information about marker
            styles.

        %(cmap_doc)s

            This parameter is ignored if *c* is RGB(A).

        %(norm_doc)s

            This parameter is ignored if *c* is RGB(A).

        %(vmin_vmax_doc)s

            This parameter is ignored if *c* is RGB(A).

        alpha : float, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        linewidths : float or array-like, default: :rc:`lines.linewidth`
            The linewidth of the marker edges. Note: The default *edgecolors*
            is 'face'. You may want to change this as well.

        edgecolors : {'face', 'none', *None*} or color or sequence of color, default: :rc:`scatter.edgecolors`
            The edge color of the marker. Possible values:

            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.

            For non-filled markers, *edgecolors* is ignored. Instead, the color
            is determined like with 'face', i.e. from *c*, *colors*, or
            *facecolors*.

        plotnonfinite : bool, default: False
            Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf``
            or ``nan``). If ``True`` the points are drawn with the *bad*
            colormap color (see `.Colormap.set_bad`).

        Returns
        -------
        `~matplotlib.collections.PathCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs : `~matplotlib.collections.Collection` properties

        See Also
        --------
        plot : To plot scatter plots when markers are identical in size and
            color.

        Notes
        -----
        * The `.plot` function will be faster for scatterplots where markers
          don't vary in size or color.

        * Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in which
          case all masks will be combined and only unmasked points will be
          plotted.

        * Fundamentally, scatter works with 1D arrays; *x*, *y*, *s*, and *c*
          may be input as N-D arrays, but within scatter they will be
          flattened. The exception is *c*, which will be flattened only if its
          size matches the size of *x* and *y*.

        """
    if edgecolors is not None:
        kwargs.update({'edgecolors': edgecolors})
    if linewidths is not None:
        kwargs.update({'linewidths': linewidths})
    kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
    linewidths = kwargs.pop('linewidth', None)
    edgecolors = kwargs.pop('edgecolor', None)
    x, y = self._process_unit_info([('x', x), ('y', y)], kwargs)
    x = np.ma.ravel(x)
    y = np.ma.ravel(y)
    if x.size != y.size:
        raise ValueError('x and y must be the same size')
    if s is None:
        s = 20 if mpl.rcParams['_internal.classic_mode'] else mpl.rcParams['lines.markersize'] ** 2.0
    s = np.ma.ravel(s)
    if len(s) not in (1, x.size) or (not np.issubdtype(s.dtype, np.floating) and (not np.issubdtype(s.dtype, np.integer))):
        raise ValueError('s must be a scalar, or float array-like with the same size as x and y')
    orig_edgecolor = edgecolors
    if edgecolors is None:
        orig_edgecolor = kwargs.get('edgecolor', None)
    c, colors, edgecolors = self._parse_scatter_color_args(c, edgecolors, kwargs, x.size, get_next_color_func=self._get_patches_for_fill.get_next_color)
    if plotnonfinite and colors is None:
        c = np.ma.masked_invalid(c)
        x, y, s, edgecolors, linewidths = cbook._combine_masks(x, y, s, edgecolors, linewidths)
    else:
        x, y, s, c, colors, edgecolors, linewidths = cbook._combine_masks(x, y, s, c, colors, edgecolors, linewidths)
    if x.size in (3, 4) and np.ma.is_masked(edgecolors) and (not np.ma.is_masked(orig_edgecolor)):
        edgecolors = edgecolors.data
    scales = s
    if marker is None:
        marker = mpl.rcParams['scatter.marker']
    if isinstance(marker, mmarkers.MarkerStyle):
        marker_obj = marker
    else:
        marker_obj = mmarkers.MarkerStyle(marker)
    path = marker_obj.get_path().transformed(marker_obj.get_transform())
    if not marker_obj.is_filled():
        if orig_edgecolor is not None:
            _api.warn_external(f'You passed a edgecolor/edgecolors ({orig_edgecolor!r}) for an unfilled marker ({marker!r}).  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.')
        if marker_obj.get_fillstyle() == 'none':
            edgecolors = colors
            colors = 'none'
        else:
            edgecolors = 'face'
        if linewidths is None:
            linewidths = mpl.rcParams['lines.linewidth']
        elif np.iterable(linewidths):
            linewidths = [lw if lw is not None else mpl.rcParams['lines.linewidth'] for lw in linewidths]
    offsets = np.ma.column_stack([x, y])
    collection = mcoll.PathCollection((path,), scales, facecolors=colors, edgecolors=edgecolors, linewidths=linewidths, offsets=offsets, offset_transform=kwargs.pop('transform', self.transData), alpha=alpha)
    collection.set_transform(mtransforms.IdentityTransform())
    if colors is None:
        collection.set_array(c)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection._scale_norm(norm, vmin, vmax)
    else:
        extra_kwargs = {'cmap': cmap, 'norm': norm, 'vmin': vmin, 'vmax': vmax}
        extra_keys = [k for k, v in extra_kwargs.items() if v is not None]
        if any(extra_keys):
            keys_str = ', '.join((f"'{k}'" for k in extra_keys))
            _api.warn_external(f"No data for colormapping provided via 'c'. Parameters {keys_str} will be ignored")
    collection._internal_update(kwargs)
    if mpl.rcParams['_internal.classic_mode']:
        if self._xmargin < 0.05 and x.size > 0:
            self.set_xmargin(0.05)
        if self._ymargin < 0.05 and x.size > 0:
            self.set_ymargin(0.05)
    self.add_collection(collection)
    self._request_autoscale_view()
    return collection