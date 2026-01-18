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
def format_xdata(self, x):
    """
        Return *x* formatted as an x-value.

        This function will use the `.fmt_xdata` attribute if it is not None,
        else will fall back on the xaxis major formatter.
        """
    return (self.fmt_xdata if self.fmt_xdata is not None else self.xaxis.get_major_formatter().format_data_short)(x)