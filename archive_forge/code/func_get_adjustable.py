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
def get_adjustable(self):
    """
        Return whether the Axes will adjust its physical dimension ('box') or
        its data limits ('datalim') to achieve the desired aspect ratio.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.
        """
    return self._adjustable