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
def minorticks_on(self):
    """
        Display minor ticks on the Axes.

        Displaying minor ticks may reduce performance; you may turn them off
        using `minorticks_off()` if drawing speed is a problem.
        """
    for ax in (self.xaxis, self.yaxis):
        scale = ax.get_scale()
        if scale == 'log':
            s = ax._scale
            ax.set_minor_locator(mticker.LogLocator(s.base, s.subs))
        elif scale == 'symlog':
            s = ax._scale
            ax.set_minor_locator(mticker.SymmetricalLogLocator(s._transform, s.subs))
        else:
            ax.set_minor_locator(mticker.AutoMinorLocator())