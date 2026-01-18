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
def get_marker_style(line):
    """Get the style dictionary for matplotlib marker objects"""
    style = {}
    style['alpha'] = line.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['facecolor'] = export_color(line.get_markerfacecolor())
    style['edgecolor'] = export_color(line.get_markeredgecolor())
    style['edgewidth'] = line.get_markeredgewidth()
    style['marker'] = line.get_marker()
    markerstyle = MarkerStyle(line.get_marker())
    markersize = line.get_markersize()
    markertransform = markerstyle.get_transform() + Affine2D().scale(markersize, -markersize)
    style['markerpath'] = SVG_path(markerstyle.get_path(), markertransform)
    style['markersize'] = markersize
    style['zorder'] = line.get_zorder()
    return style