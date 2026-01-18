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
def export_color(color):
    """Convert matplotlib color code to hex color or RGBA color"""
    if color is None or colorConverter.to_rgba(color)[3] == 0:
        return 'none'
    elif colorConverter.to_rgba(color)[3] == 1:
        rgb = colorConverter.to_rgb(color)
        return '#{0:02X}{1:02X}{2:02X}'.format(*(int(255 * c) for c in rgb))
    else:
        c = colorConverter.to_rgba(color)
        return 'rgba(' + ', '.join((str(int(np.round(val * 255))) for val in c[:3])) + ', ' + str(c[3]) + ')'