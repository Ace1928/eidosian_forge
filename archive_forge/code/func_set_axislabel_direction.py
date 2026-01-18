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
def set_axislabel_direction(self, label_direction):
    """
        Adjust the direction of the axis label.

        Note that the *label_direction*\\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        label_direction : {"+", "-"}
        """
    self._axislabel_add_angle = _api.check_getitem({'+': 0, '-': 180}, label_direction=label_direction)