import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def get_grid_style(axis):
    gridlines = axis.get_gridlines()
    if axis._major_tick_kw['gridOn'] and len(gridlines) > 0:
        color = export_color(gridlines[0].get_color())
        alpha = gridlines[0].get_alpha()
        dasharray = get_dasharray(gridlines[0])
        return dict(gridOn=True, color=color, dasharray=dasharray, alpha=alpha)
    else:
        return {'gridOn': False}