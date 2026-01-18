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
def relim(self, visible_only=False):
    """
        Recompute the data limits based on current artists.

        At present, `.Collection` instances are not supported.

        Parameters
        ----------
        visible_only : bool, default: False
            Whether to exclude invisible artists.
        """
    self.dataLim.ignore(True)
    self.dataLim.set_points(mtransforms.Bbox.null().get_points())
    self.ignore_existing_data_limits = True
    for artist in self._children:
        if not visible_only or artist.get_visible():
            if isinstance(artist, mlines.Line2D):
                self._update_line_limits(artist)
            elif isinstance(artist, mpatches.Patch):
                self._update_patch_limits(artist)
            elif isinstance(artist, mimage.AxesImage):
                self._update_image_limits(artist)