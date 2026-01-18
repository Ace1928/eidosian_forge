import functools
from itertools import chain
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform
from .axislines import (
from .axis_artist import AxisArtist
from .grid_finder import GridFinder
def get_tick_iterator(self, nth_coord, axis_side, minor=False):
    angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
    lon_or_lat = ['lon', 'lat'][nth_coord]
    if not minor:
        for (xy, a), l in zip(self._grid_info[lon_or_lat]['tick_locs'][axis_side], self._grid_info[lon_or_lat]['tick_labels'][axis_side]):
            angle_normal = a
            yield (xy, angle_normal, angle_tangent, l)
    else:
        for (xy, a), l in zip(self._grid_info[lon_or_lat]['tick_locs'][axis_side], self._grid_info[lon_or_lat]['tick_labels'][axis_side]):
            angle_normal = a
            yield (xy, angle_normal, angle_tangent, '')