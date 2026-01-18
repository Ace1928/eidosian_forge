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
def set_axis_direction(self, axis_direction):
    """
        Adjust the direction, text angle, and text alignment of tick labels
        and axis labels following the Matplotlib convention for the rectangle
        axes.

        The *axis_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel direction      "-"        "+"       "+"        "-"
        axislabel direction      "-"        "+"       "+"        "-"
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the direction "+" and "-" are relative to the direction of
        the increasing coordinate. Also, the text angles are actually
        relative to (90 + angle of the direction to the ticklabel),
        which gives 0 for bottom axis.

        Parameters
        ----------
        axis_direction : {"left", "bottom", "right", "top"}
        """
    self.major_ticklabels.set_axis_direction(axis_direction)
    self.label.set_axis_direction(axis_direction)
    self._axis_direction = axis_direction
    if axis_direction in ['left', 'top']:
        self.set_ticklabel_direction('-')
        self.set_axislabel_direction('-')
    else:
        self.set_ticklabel_direction('+')
        self.set_axislabel_direction('+')