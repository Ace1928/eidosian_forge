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
def add_line(self, line):
    """
        Add a `.Line2D` to the Axes; return the line.
        """
    _api.check_isinstance(mlines.Line2D, line=line)
    self._set_artist_props(line)
    if line.get_clip_path() is None:
        line.set_clip_path(self.patch)
    self._update_line_limits(line)
    if not line.get_label():
        line.set_label(f'_child{len(self._children)}')
    self._children.append(line)
    line._remove_method = self._children.remove
    self.stale = True
    return line