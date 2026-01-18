import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def make_axes_area_auto_adjustable(ax, use_axes=None, pad=0.1, adjust_dirs=None):
    """
    Add auto-adjustable padding around *ax* to take its decorations (title,
    labels, ticks, ticklabels) into account during layout, using
    `.Divider.add_auto_adjustable_area`.

    By default, padding is determined from the decorations of *ax*.
    Pass *use_axes* to consider the decorations of other Axes instead.
    """
    if adjust_dirs is None:
        adjust_dirs = ['left', 'right', 'bottom', 'top']
    divider = make_axes_locatable(ax)
    if use_axes is None:
        use_axes = ax
    divider.add_auto_adjustable_area(use_axes=use_axes, pad=pad, adjust_dirs=adjust_dirs)