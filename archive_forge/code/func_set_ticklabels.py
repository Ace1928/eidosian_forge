import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
@_api.rename_parameter('3.7', 'ticklabels', 'labels')
def set_ticklabels(self, labels, *, minor=False, fontdict=None, **kwargs):
    """
        [*Discouraged*] Set this Axis' tick labels with list of string labels.

        .. admonition:: Discouraged

            The use of this method is discouraged, because of the dependency on
            tick positions. In most cases, you'll want to use
            ``Axes.set_[x/y/z]ticks(positions, labels)`` or ``Axis.set_ticks``
            instead.

            If you are using this method, you should always fix the tick
            positions before, e.g. by using `.Axis.set_ticks` or by explicitly
            setting a `~.ticker.FixedLocator`. Otherwise, ticks are free to
            move and the labels may end up in unexpected positions.

        Parameters
        ----------
        labels : sequence of str or of `.Text`\\s
            Texts for labeling each tick location in the sequence set by
            `.Axis.set_ticks`; the number of labels must match the number of
            locations.

        minor : bool
            If True, set minor ticks instead of major ticks.

        fontdict : dict, optional

            .. admonition:: Discouraged

               The use of *fontdict* is discouraged. Parameters should be passed as
               individual keyword arguments or using dictionary-unpacking
               ``set_ticklabels(..., **fontdict)``.

            A dictionary controlling the appearance of the ticklabels.
            The default *fontdict* is::

               {'fontsize': rcParams['axes.titlesize'],
                'fontweight': rcParams['axes.titleweight'],
                'verticalalignment': 'baseline',
                'horizontalalignment': loc}

        **kwargs
            Text properties.

            .. warning::

                This only sets the properties of the current ticks.
                Ticks are not guaranteed to be persistent. Various operations
                can create, delete and modify the Tick instances. There is an
                imminent risk that these settings can get lost if you work on
                the figure further (including also panning/zooming on a
                displayed figure).

                Use `.set_tick_params` instead if possible.

        Returns
        -------
        list of `.Text`\\s
            For each tick, includes ``tick.label1`` if it is visible, then
            ``tick.label2`` if it is visible, in that order.
        """
    try:
        labels = [t.get_text() if hasattr(t, 'get_text') else t for t in labels]
    except TypeError:
        raise TypeError(f'{labels:=} must be a sequence') from None
    locator = self.get_minor_locator() if minor else self.get_major_locator()
    if not labels:
        formatter = mticker.NullFormatter()
    elif isinstance(locator, mticker.FixedLocator):
        if len(locator.locs) != len(labels) and len(labels) != 0:
            raise ValueError(f'The number of FixedLocator locations ({len(locator.locs)}), usually from a call to set_ticks, does not match the number of labels ({len(labels)}).')
        tickd = {loc: lab for loc, lab in zip(locator.locs, labels)}
        func = functools.partial(self._format_with_dict, tickd)
        formatter = mticker.FuncFormatter(func)
    else:
        _api.warn_external('set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.')
        formatter = mticker.FixedFormatter(labels)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='FixedFormatter should only be used together with FixedLocator')
        if minor:
            self.set_minor_formatter(formatter)
            locs = self.get_minorticklocs()
            ticks = self.get_minor_ticks(len(locs))
        else:
            self.set_major_formatter(formatter)
            locs = self.get_majorticklocs()
            ticks = self.get_major_ticks(len(locs))
    ret = []
    if fontdict is not None:
        kwargs.update(fontdict)
    for pos, (loc, tick) in enumerate(zip(locs, ticks)):
        tick.update_position(loc)
        tick_label = formatter(loc, pos)
        tick.label1.set_text(tick_label)
        tick.label1._internal_update(kwargs)
        tick.label2.set_text(tick_label)
        tick.label2._internal_update(kwargs)
        if tick.label1.get_visible():
            ret.append(tick.label1)
        if tick.label2.get_visible():
            ret.append(tick.label2)
    self.stale = True
    return ret