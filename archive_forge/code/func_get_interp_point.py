import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
def get_interp_point(idx):
    im1 = max(idx - 1, 0)
    ind_values = ind[im1:idx + 1]
    diff_values = dep1[im1:idx + 1] - dep2[im1:idx + 1]
    dep1_values = dep1[im1:idx + 1]
    if len(diff_values) == 2:
        if np.ma.is_masked(diff_values[1]):
            return (ind[im1], dep1[im1])
        elif np.ma.is_masked(diff_values[0]):
            return (ind[idx], dep1[idx])
    diff_order = diff_values.argsort()
    diff_root_ind = np.interp(0, diff_values[diff_order], ind_values[diff_order])
    ind_order = ind_values.argsort()
    diff_root_dep = np.interp(diff_root_ind, ind_values[ind_order], dep1_values[ind_order])
    return (diff_root_ind, diff_root_dep)