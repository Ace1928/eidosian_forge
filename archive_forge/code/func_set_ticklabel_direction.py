from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def set_ticklabel_direction(self, tick_direction):
    """
        Adjust the direction of the tick labels.

        Note that the *tick_direction*\\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        tick_direction : {"+", "-"}
        """
    self._ticklabel_add_angle = _api.check_getitem({'+': 0, '-': 180}, tick_direction=tick_direction)