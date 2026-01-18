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
def bxp(self, bxpstats, positions=None, widths=None, vert=True, patch_artist=False, shownotches=False, showmeans=False, showcaps=True, showbox=True, showfliers=True, boxprops=None, whiskerprops=None, flierprops=None, medianprops=None, capprops=None, meanprops=None, meanline=False, manage_ticks=True, zorder=None, capwidths=None):
    """
        Drawing function for box and whisker plots.

        Make a box and whisker plot for each column of *x* or each
        vector in sequence *x*.  The box extends from the lower to
        upper quartile values of the data, with a line at the median.
        The whiskers extend from the box to show the range of the
        data.  Flier points are those past the end of the whiskers.

        Parameters
        ----------
        bxpstats : list of dicts
          A list of dictionaries containing stats for each boxplot.
          Required keys are:

          - ``med``: Median (scalar).
          - ``q1``, ``q3``: First & third quartiles (scalars).
          - ``whislo``, ``whishi``: Lower & upper whisker positions (scalars).

          Optional keys are:

          - ``mean``: Mean (scalar).  Needed if ``showmeans=True``.
          - ``fliers``: Data beyond the whiskers (array-like).
            Needed if ``showfliers=True``.
          - ``cilo``, ``cihi``: Lower & upper confidence intervals
            about the median. Needed if ``shownotches=True``.
          - ``label``: Name of the dataset (str).  If available,
            this will be used a tick label for the boxplot

        positions : array-like, default: [1, 2, ..., n]
          The positions of the boxes. The ticks and limits
          are automatically set to match the positions.

        widths : float or array-like, default: None
          The widths of the boxes.  The default is
          ``clip(0.15*(distance between extreme positions), 0.15, 0.5)``.

        capwidths : float or array-like, default: None
          Either a scalar or a vector and sets the width of each cap.
          The default is ``0.5*(width of the box)``, see *widths*.

        vert : bool, default: True
          If `True` (default), makes the boxes vertical.
          If `False`, makes horizontal boxes.

        patch_artist : bool, default: False
          If `False` produces boxes with the `.Line2D` artist.
          If `True` produces boxes with the `~matplotlib.patches.Patch` artist.

        shownotches, showmeans, showcaps, showbox, showfliers : bool
          Whether to draw the CI notches, the mean value (both default to
          False), the caps, the box, and the fliers (all three default to
          True).

        boxprops, whiskerprops, capprops, flierprops, medianprops, meanprops : dict, optional
          Artist properties for the boxes, whiskers, caps, fliers, medians, and
          means.

        meanline : bool, default: False
          If `True` (and *showmeans* is `True`), will try to render the mean
          as a line spanning the full width of the box according to
          *meanprops*. Not recommended if *shownotches* is also True.
          Otherwise, means will be shown as points.

        manage_ticks : bool, default: True
          If True, the tick locations and labels will be adjusted to match the
          boxplot positions.

        zorder : float, default: ``Line2D.zorder = 2``
          The zorder of the resulting boxplot.

        Returns
        -------
        dict
          A dictionary mapping each component of the boxplot to a list
          of the `.Line2D` instances created. That dictionary has the
          following keys (assuming vertical boxplots):

          - ``boxes``: main bodies of the boxplot showing the quartiles, and
            the median's confidence intervals if enabled.
          - ``medians``: horizontal lines at the median of each box.
          - ``whiskers``: vertical lines up to the last non-outlier data.
          - ``caps``: horizontal lines at the ends of the whiskers.
          - ``fliers``: points representing data beyond the whiskers (fliers).
          - ``means``: points or lines representing the means.

        Examples
        --------
        .. plot:: gallery/statistics/bxp.py
        """
    medianprops = {'solid_capstyle': 'butt', 'dash_capstyle': 'butt', **(medianprops or {})}
    meanprops = {'solid_capstyle': 'butt', 'dash_capstyle': 'butt', **(meanprops or {})}
    whiskers = []
    caps = []
    boxes = []
    medians = []
    means = []
    fliers = []
    datalabels = []
    if zorder is None:
        zorder = mlines.Line2D.zorder
    zdelta = 0.1

    def merge_kw_rc(subkey, explicit, zdelta=0, usemarker=True):
        d = {k.split('.')[-1]: v for k, v in mpl.rcParams.items() if k.startswith(f'boxplot.{subkey}props')}
        d['zorder'] = zorder + zdelta
        if not usemarker:
            d['marker'] = ''
        d.update(cbook.normalize_kwargs(explicit, mlines.Line2D))
        return d
    box_kw = {'linestyle': mpl.rcParams['boxplot.boxprops.linestyle'], 'linewidth': mpl.rcParams['boxplot.boxprops.linewidth'], 'edgecolor': mpl.rcParams['boxplot.boxprops.color'], 'facecolor': 'white' if mpl.rcParams['_internal.classic_mode'] else mpl.rcParams['patch.facecolor'], 'zorder': zorder, **cbook.normalize_kwargs(boxprops, mpatches.PathPatch)} if patch_artist else merge_kw_rc('box', boxprops, usemarker=False)
    whisker_kw = merge_kw_rc('whisker', whiskerprops, usemarker=False)
    cap_kw = merge_kw_rc('cap', capprops, usemarker=False)
    flier_kw = merge_kw_rc('flier', flierprops)
    median_kw = merge_kw_rc('median', medianprops, zdelta, usemarker=False)
    mean_kw = merge_kw_rc('mean', meanprops, zdelta)
    removed_prop = 'marker' if meanline else 'linestyle'
    if meanprops is None or removed_prop not in meanprops:
        mean_kw[removed_prop] = ''
    maybe_swap = slice(None) if vert else slice(None, None, -1)

    def do_plot(xs, ys, **kwargs):
        return self.plot(*[xs, ys][maybe_swap], **kwargs)[0]

    def do_patch(xs, ys, **kwargs):
        path = mpath.Path._create_closed(np.column_stack([xs, ys][maybe_swap]))
        patch = mpatches.PathPatch(path, **kwargs)
        self.add_artist(patch)
        return patch
    N = len(bxpstats)
    datashape_message = 'List of boxplot statistics and `{0}` values must have same the length'
    if positions is None:
        positions = list(range(1, N + 1))
    elif len(positions) != N:
        raise ValueError(datashape_message.format('positions'))
    positions = np.array(positions)
    if len(positions) > 0 and (not all((isinstance(p, Real) for p in positions))):
        raise TypeError('positions should be an iterable of numbers')
    if widths is None:
        widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
    elif np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format('widths'))
    if capwidths is None:
        capwidths = 0.5 * np.array(widths)
    elif np.isscalar(capwidths):
        capwidths = [capwidths] * N
    elif len(capwidths) != N:
        raise ValueError(datashape_message.format('capwidths'))
    for pos, width, stats, capwidth in zip(positions, widths, bxpstats, capwidths):
        datalabels.append(stats.get('label', pos))
        whis_x = [pos, pos]
        whislo_y = [stats['q1'], stats['whislo']]
        whishi_y = [stats['q3'], stats['whishi']]
        cap_left = pos - capwidth * 0.5
        cap_right = pos + capwidth * 0.5
        cap_x = [cap_left, cap_right]
        cap_lo = np.full(2, stats['whislo'])
        cap_hi = np.full(2, stats['whishi'])
        box_left = pos - width * 0.5
        box_right = pos + width * 0.5
        med_y = [stats['med'], stats['med']]
        if shownotches:
            notch_left = pos - width * 0.25
            notch_right = pos + width * 0.25
            box_x = [box_left, box_right, box_right, notch_right, box_right, box_right, box_left, box_left, notch_left, box_left, box_left]
            box_y = [stats['q1'], stats['q1'], stats['cilo'], stats['med'], stats['cihi'], stats['q3'], stats['q3'], stats['cihi'], stats['med'], stats['cilo'], stats['q1']]
            med_x = [notch_left, notch_right]
        else:
            box_x = [box_left, box_right, box_right, box_left, box_left]
            box_y = [stats['q1'], stats['q1'], stats['q3'], stats['q3'], stats['q1']]
            med_x = [box_left, box_right]
        if showbox:
            do_box = do_patch if patch_artist else do_plot
            boxes.append(do_box(box_x, box_y, **box_kw))
        whiskers.append(do_plot(whis_x, whislo_y, **whisker_kw))
        whiskers.append(do_plot(whis_x, whishi_y, **whisker_kw))
        if showcaps:
            caps.append(do_plot(cap_x, cap_lo, **cap_kw))
            caps.append(do_plot(cap_x, cap_hi, **cap_kw))
        medians.append(do_plot(med_x, med_y, **median_kw))
        if showmeans:
            if meanline:
                means.append(do_plot([box_left, box_right], [stats['mean'], stats['mean']], **mean_kw))
            else:
                means.append(do_plot([pos], [stats['mean']], **mean_kw))
        if showfliers:
            flier_x = np.full(len(stats['fliers']), pos, dtype=np.float64)
            flier_y = stats['fliers']
            fliers.append(do_plot(flier_x, flier_y, **flier_kw))
    if manage_ticks:
        axis_name = 'x' if vert else 'y'
        interval = getattr(self.dataLim, f'interval{axis_name}')
        axis = self._axis_map[axis_name]
        positions = axis.convert_units(positions)
        interval[:] = (min(interval[0], min(positions) - 0.5), max(interval[1], max(positions) + 0.5))
        for median, position in zip(medians, positions):
            getattr(median.sticky_edges, axis_name).extend([position - 0.5, position + 0.5])
        locator = axis.get_major_locator()
        if not isinstance(axis.get_major_locator(), mticker.FixedLocator):
            locator = mticker.FixedLocator([])
            axis.set_major_locator(locator)
        locator.locs = np.array([*locator.locs, *positions])
        formatter = axis.get_major_formatter()
        if not isinstance(axis.get_major_formatter(), mticker.FixedFormatter):
            formatter = mticker.FixedFormatter([])
            axis.set_major_formatter(formatter)
        formatter.seq = [*formatter.seq, *datalabels]
        self._request_autoscale_view()
    return dict(whiskers=whiskers, caps=caps, boxes=boxes, medians=medians, fliers=fliers, means=means)