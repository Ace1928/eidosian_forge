from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
def get_gridlines(self, which='major', axis='both'):
    """
        Return list of gridline coordinates in data coordinates.

        Parameters
        ----------
        which : {"both", "major", "minor"}
        axis : {"both", "x", "y"}
        """
    _api.check_in_list(['both', 'major', 'minor'], which=which)
    _api.check_in_list(['both', 'x', 'y'], axis=axis)
    gridlines = []
    if axis in ('both', 'x'):
        locs = []
        y1, y2 = self.axes.get_ylim()
        if which in ('both', 'major'):
            locs.extend(self.axes.xaxis.major.locator())
        if which in ('both', 'minor'):
            locs.extend(self.axes.xaxis.minor.locator())
        for x in locs:
            gridlines.append([[x, x], [y1, y2]])
    if axis in ('both', 'y'):
        x1, x2 = self.axes.get_xlim()
        locs = []
        if self.axes.yaxis._major_tick_kw['gridOn']:
            locs.extend(self.axes.yaxis.major.locator())
        if self.axes.yaxis._minor_tick_kw['gridOn']:
            locs.extend(self.axes.yaxis.minor.locator())
        for y in locs:
            gridlines.append([[x1, x2], [y, y]])
    return gridlines