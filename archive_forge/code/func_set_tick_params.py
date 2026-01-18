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
def set_tick_params(self, which='major', reset=False, **kwargs):
    """
        Set appearance parameters for ticks, ticklabels, and gridlines.

        For documentation of keyword arguments, see
        :meth:`matplotlib.axes.Axes.tick_params`.

        See Also
        --------
        .Axis.get_tick_params
            View the current style settings for ticks, ticklabels, and
            gridlines.
        """
    _api.check_in_list(['major', 'minor', 'both'], which=which)
    kwtrans = self._translate_tick_params(kwargs)
    if reset:
        if which in ['major', 'both']:
            self._reset_major_tick_kw()
            self._major_tick_kw.update(kwtrans)
        if which in ['minor', 'both']:
            self._reset_minor_tick_kw()
            self._minor_tick_kw.update(kwtrans)
        self.reset_ticks()
    else:
        if which in ['major', 'both']:
            self._major_tick_kw.update(kwtrans)
            for tick in self.majorTicks:
                tick._apply_params(**kwtrans)
        if which in ['minor', 'both']:
            self._minor_tick_kw.update(kwtrans)
            for tick in self.minorTicks:
                tick._apply_params(**kwtrans)
        if 'label1On' in kwtrans or 'label2On' in kwtrans:
            self.offsetText.set_visible(self._major_tick_kw.get('label1On', False) or self._major_tick_kw.get('label2On', False))
        if 'labelcolor' in kwtrans:
            self.offsetText.set_color(kwtrans['labelcolor'])
    self.stale = True