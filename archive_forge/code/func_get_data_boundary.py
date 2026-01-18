import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple
@_api.deprecated('3.8')
def get_data_boundary(self, side):
    """
        Return v=0, nth=1.
        """
    lon1, lon2, lat1, lat2 = self.grid_finder.extreme_finder(*[None] * 5)
    return dict(left=(lon1, 0), right=(lon2, 0), bottom=(lat1, 1), top=(lat2, 1))[side]