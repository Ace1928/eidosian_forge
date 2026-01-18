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
def get_proj(self):
    """Create the projection matrix from the current viewing position."""
    box_aspect = self._roll_to_vertical(self._box_aspect)
    worldM = proj3d.world_transformation(*self.get_xlim3d(), *self.get_ylim3d(), *self.get_zlim3d(), pb_aspect=box_aspect)
    R = 0.5 * box_aspect
    elev_rad = np.deg2rad(self.elev)
    azim_rad = np.deg2rad(self.azim)
    p0 = np.cos(elev_rad) * np.cos(azim_rad)
    p1 = np.cos(elev_rad) * np.sin(azim_rad)
    p2 = np.sin(elev_rad)
    ps = self._roll_to_vertical([p0, p1, p2])
    eye = R + self._dist * ps
    vvec = R - eye
    self._eye = eye
    self._vvec = vvec / np.linalg.norm(vvec)
    u, v, w = self._calc_view_axes(eye)
    self._view_u = u
    self._view_v = v
    self._view_w = w
    if self._focal_length == np.inf:
        viewM = proj3d._view_transformation_uvw(u, v, w, eye)
        projM = proj3d._ortho_transformation(-self._dist, self._dist)
    else:
        eye_focal = R + self._dist * ps * self._focal_length
        viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
        projM = proj3d._persp_transformation(-self._dist, self._dist, self._focal_length)
    M0 = np.dot(viewM, worldM)
    M = np.dot(projM, M0)
    return M