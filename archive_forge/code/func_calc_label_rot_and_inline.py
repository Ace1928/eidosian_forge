from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
@_api.deprecated('3.8')
def calc_label_rot_and_inline(self, slc, ind, lw, lc=None, spacing=5):
    """
        Calculate the appropriate label rotation given the linecontour
        coordinates in screen units, the index of the label location and the
        label width.

        If *lc* is not None or empty, also break contours and compute
        inlining.

        *spacing* is the empty space to leave around the label, in pixels.

        Both tasks are done together to avoid calculating path lengths
        multiple times, which is relatively costly.

        The method used here involves computing the path length along the
        contour in pixel coordinates and then looking approximately (label
        width / 2) away from central point to determine rotation and then to
        break contour if desired.
        """
    if lc is None:
        lc = []
    hlw = lw / 2.0
    closed = _is_closed_polygon(slc)
    if closed:
        slc = np.concatenate([slc[ind:-1], slc[:ind + 1]])
        if len(lc):
            lc = np.concatenate([lc[ind:-1], lc[:ind + 1]])
        ind = 0
    pl = np.zeros(slc.shape[0], dtype=float)
    dx = np.diff(slc, axis=0)
    pl[1:] = np.cumsum(np.hypot(dx[:, 0], dx[:, 1]))
    pl = pl - pl[ind]
    xi = np.array([-hlw, hlw])
    if closed:
        dp = np.array([pl[-1], 0])
    else:
        dp = np.zeros_like(xi)
    (dx,), (dy,) = (np.diff(np.interp(dp + xi, pl, slc_col)) for slc_col in slc.T)
    rotation = np.rad2deg(np.arctan2(dy, dx))
    if self.rightside_up:
        rotation = (rotation + 90) % 180 - 90
    nlc = []
    if len(lc):
        xi = dp + xi + np.array([-spacing, spacing])
        I = np.interp(xi, pl, np.arange(len(pl)), left=-1, right=-1)
        I = [np.floor(I[0]).astype(int), np.ceil(I[1]).astype(int)]
        if I[0] != -1:
            xy1 = [np.interp(xi[0], pl, lc_col) for lc_col in lc.T]
        if I[1] != -1:
            xy2 = [np.interp(xi[1], pl, lc_col) for lc_col in lc.T]
        if closed:
            if all((i != -1 for i in I)):
                nlc.append(np.vstack([xy2, lc[I[1]:I[0] + 1], xy1]))
        else:
            if I[0] != -1:
                nlc.append(np.vstack([lc[:I[0] + 1], xy1]))
            if I[1] != -1:
                nlc.append(np.vstack([xy2, lc[I[1]:]]))
    return (rotation, nlc)