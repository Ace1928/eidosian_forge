from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
    """
        Set the label for the y-axis.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *y* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
    if labelpad is not None:
        self.yaxis.labelpad = labelpad
    protected_kw = ['y', 'horizontalalignment', 'ha']
    if {*kwargs} & {*protected_kw}:
        if loc is not None:
            raise TypeError(f"Specifying 'loc' is disallowed when any of its corresponding low level keyword arguments ({protected_kw}) are also supplied")
    else:
        loc = loc if loc is not None else mpl.rcParams['yaxis.labellocation']
        _api.check_in_list(('bottom', 'center', 'top'), loc=loc)
        y, ha = {'bottom': (0, 'left'), 'center': (0.5, 'center'), 'top': (1, 'right')}[loc]
        kwargs.update(y=y, horizontalalignment=ha)
    return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)