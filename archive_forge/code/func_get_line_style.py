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
def get_line_style(line):
    """Get the style dictionary for matplotlib line objects"""
    style = {}
    style['alpha'] = line.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['color'] = export_color(line.get_color())
    style['linewidth'] = line.get_linewidth()
    style['dasharray'] = get_dasharray(line)
    style['zorder'] = line.get_zorder()
    style['drawstyle'] = line.get_drawstyle()
    return style