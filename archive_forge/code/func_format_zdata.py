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
def format_zdata(self, z):
    """
        Return *z* string formatted.  This function will use the
        :attr:`fmt_zdata` attribute if it is callable, else will fall
        back on the zaxis major formatter
        """
    try:
        return self.fmt_zdata(z)
    except (AttributeError, TypeError):
        func = self.zaxis.get_major_formatter().format_data_short
        val = func(z)
        return val