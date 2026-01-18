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
def drag_pan(self, button, key, x, y):
    p = self._pan_start
    (xdata, ydata), (xdata_start, ydata_start) = p.trans_inverse.transform([(x, y), (p.x, p.y)])
    self._sx, self._sy = (xdata, ydata)
    self.start_pan(x, y, button)
    du, dv = (xdata - xdata_start, ydata - ydata_start)
    dw = 0
    if key == 'x':
        dv = 0
    elif key == 'y':
        du = 0
    if du == 0 and dv == 0:
        return
    R = np.array([self._view_u, self._view_v, self._view_w])
    R = -R / self._box_aspect * self._dist
    duvw_projected = R.T @ np.array([du, dv, dw])
    minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
    dx = (maxx - minx) * duvw_projected[0]
    dy = (maxy - miny) * duvw_projected[1]
    dz = (maxz - minz) * duvw_projected[2]
    self.set_xlim3d(minx + dx, maxx + dx)
    self.set_ylim3d(miny + dy, maxy + dy)
    self.set_zlim3d(minz + dz, maxz + dz)