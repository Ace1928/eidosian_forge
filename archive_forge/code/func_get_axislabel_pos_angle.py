from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def get_axislabel_pos_angle(self, axes):
    """
        Return the label reference position in transAxes.

        get_label_transform() returns a transform of (transAxes+offset)
        """
    angle = [0, 90][self.nth_coord]
    fixed_coord = 1 - self.nth_coord
    data_to_axes = axes.transData - axes.transAxes
    p = data_to_axes.transform([self._value, self._value])
    verts = self._to_xy(0.5, const=p[fixed_coord])
    if 0 <= verts[fixed_coord] <= 1:
        return (verts, angle)
    else:
        return (None, None)