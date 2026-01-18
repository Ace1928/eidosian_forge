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
def format_coord(self, xv, yv, renderer=None):
    """
        Return a string giving the current view rotation angles, or the x, y, z
        coordinates of the point on the nearest axis pane underneath the mouse
        cursor, depending on the mouse button pressed.
        """
    coords = ''
    if self.button_pressed in self._rotate_btn:
        coords = self._rotation_coords()
    elif self.M is not None:
        coords = self._location_coords(xv, yv, renderer)
    return coords