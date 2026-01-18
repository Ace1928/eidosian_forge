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
def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
    """
        Set a title for the Axes.

        Set one of the three available Axes titles. The available titles
        are positioned above the Axes in the center, flush with the left
        edge, and flush with the right edge.

        Parameters
        ----------
        label : str
            Text to use for the title

        fontdict : dict

            .. admonition:: Discouraged

               The use of *fontdict* is discouraged. Parameters should be passed as
               individual keyword arguments or using dictionary-unpacking
               ``set_title(..., **fontdict)``.

            A dictionary controlling the appearance of the title text,
            the default *fontdict* is::

               {'fontsize': rcParams['axes.titlesize'],
                'fontweight': rcParams['axes.titleweight'],
                'color': rcParams['axes.titlecolor'],
                'verticalalignment': 'baseline',
                'horizontalalignment': loc}

        loc : {'center', 'left', 'right'}, default: :rc:`axes.titlelocation`
            Which title to set.

        y : float, default: :rc:`axes.titley`
            Vertical Axes location for the title (1.0 is the top).  If
            None (the default) and :rc:`axes.titley` is also None, y is
            determined automatically to avoid decorators on the Axes.

        pad : float, default: :rc:`axes.titlepad`
            The offset of the title from the top of the Axes, in points.

        Returns
        -------
        `.Text`
            The matplotlib text instance representing the title

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            Other keyword arguments are text properties, see `.Text` for a list
            of valid text properties.
        """
    if loc is None:
        loc = mpl.rcParams['axes.titlelocation']
    if y is None:
        y = mpl.rcParams['axes.titley']
    if y is None:
        y = 1.0
    else:
        self._autotitlepos = False
    kwargs['y'] = y
    titles = {'left': self._left_title, 'center': self.title, 'right': self._right_title}
    title = _api.check_getitem(titles, loc=loc.lower())
    default = {'fontsize': mpl.rcParams['axes.titlesize'], 'fontweight': mpl.rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': loc.lower()}
    titlecolor = mpl.rcParams['axes.titlecolor']
    if not cbook._str_lower_equal(titlecolor, 'auto'):
        default['color'] = titlecolor
    if pad is None:
        pad = mpl.rcParams['axes.titlepad']
    self._set_title_offset_trans(float(pad))
    title.set_text(label)
    title.update(default)
    if fontdict is not None:
        title.update(fontdict)
    title._internal_update(kwargs)
    return title