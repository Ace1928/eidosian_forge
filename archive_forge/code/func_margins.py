from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def margins(self, *margins, x=None, y=None, z=None, tight=True):
    """
        Set or retrieve autoscaling margins.

        See `.Axes.margins` for full documentation.  Because this function
        applies to 3D Axes, it also takes a *z* argument, and returns
        ``(xmargin, ymargin, zmargin)``.
        """
    if margins and (x is not None or y is not None or z is not None):
        raise TypeError('Cannot pass both positional and keyword arguments for x, y, and/or z.')
    elif len(margins) == 1:
        x = y = z = margins[0]
    elif len(margins) == 3:
        x, y, z = margins
    elif margins:
        raise TypeError('Must pass a single positional argument for all margins, or one for each margin (x, y, z).')
    if x is None and y is None and (z is None):
        if tight is not True:
            _api.warn_external(f'ignoring tight={tight!r} in get mode')
        return (self._xmargin, self._ymargin, self._zmargin)
    if x is not None:
        self.set_xmargin(x)
    if y is not None:
        self.set_ymargin(y)
    if z is not None:
        self.set_zmargin(z)
    self.autoscale_view(tight=tight, scalex=x is not None, scaley=y is not None, scalez=z is not None)