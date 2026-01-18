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
def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False, zmin=None, zmax=None):
    """
        Set 3D z limits.

        See `.Axes.set_ylim` for full documentation
        """
    if top is None and np.iterable(bottom):
        bottom, top = bottom
    if zmin is not None:
        if bottom is not None:
            raise TypeError("Cannot pass both 'bottom' and 'zmin'")
        bottom = zmin
    if zmax is not None:
        if top is not None:
            raise TypeError("Cannot pass both 'top' and 'zmax'")
        top = zmax
    return self.zaxis._set_lim(bottom, top, emit=emit, auto=auto)