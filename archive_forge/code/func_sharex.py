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
def sharex(self, other):
    """
        Share the x-axis with *other*.

        This is equivalent to passing ``sharex=other`` when constructing the
        Axes, and cannot be used if the x-axis is already being shared with
        another Axes.
        """
    _api.check_isinstance(_AxesBase, other=other)
    if self._sharex is not None and other is not self._sharex:
        raise ValueError('x-axis is already shared')
    self._shared_axes['x'].join(self, other)
    self._sharex = other
    self.xaxis.major = other.xaxis.major
    self.xaxis.minor = other.xaxis.minor
    x0, x1 = other.get_xlim()
    self.set_xlim(x0, x1, emit=False, auto=other.get_autoscalex_on())
    self.xaxis._scale = other.xaxis._scale