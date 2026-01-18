from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def selector_default(output_widget=None):
    """Capture selection events from the current figure, and apply the selections to Scatter objects.

    Example:

    >>> import ipyvolume as ipv
    >>> ipv.figure()
    >>> ipv.examples.gaussian()
    >>> ipv.selector_default()
    >>> ipv.show()

    Now hold the control key to do selections, type

      * 'C' for circle
      * 'R' for rectangle
      * 'L' for lasso
      * '=' for replace mode
      * '&' for logically and mode
      * '|' for logically or mode
      * '-' for subtract mode

    """
    fig = gcf()
    if output_widget is None:
        output_widget = ipywidgets.Output()
        display(output_widget)

    def lasso(data, other=None, fig=fig):
        with output_widget:
            inside = None
            if data['device'] and data['type'] == 'lasso':
                region = shapely.geometry.Polygon(data['device'])

                @np.vectorize
                def inside_polygon(x, y):
                    return region.contains(shapely.geometry.Point([x, y]))
                inside = inside_polygon
            if data['device'] and data['type'] == 'circle':
                x1, y1 = data['device']['begin']
                x2, y2 = data['device']['end']
                dx = x2 - x1
                dy = y2 - y1
                r = (dx ** 2 + dy ** 2) ** 0.5

                def inside_circle(x, y):
                    return (x - x1) ** 2 + (y - y1) ** 2 < r ** 2
                inside = inside_circle
            if data['device'] and data['type'] == 'rectangle':
                x1, y1 = data['device']['begin']
                x2, y2 = data['device']['end']
                x = [x1, x2]
                y = [y1, y2]
                xmin, xmax = (min(x), max(x))
                ymin, ymax = (min(y), max(y))

                def inside_rectangle(x, y):
                    return (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
                inside = inside_rectangle

            def join(x, y, mode):
                Nx = 0 if x is None or len(x[0]) == 0 else np.max(x)
                Ny = 0 if len(y[0]) == 0 else np.max(y)
                N = max(Nx, Ny)
                xmask = np.zeros(N + 1, np.bool)
                ymask = np.zeros(N + 1, np.bool)
                if x is not None:
                    xmask[x] = True
                ymask[y] = True
                if mode == 'replace':
                    return np.where(ymask)
                if mode == 'and':
                    mask = xmask & ymask
                    return np.where(ymask if x is None else mask)
                if mode == 'or':
                    mask = xmask | ymask
                    return np.where(ymask if x is None else mask)
                if mode == 'subtract':
                    mask = xmask & ~ymask
                    return np.where(ymask if x is None else mask)
            for scatter in fig.scatters:
                x, y = fig.project(scatter.x, scatter.y, scatter.z)
                mask = inside(x, y)
                scatter.selected = join(scatter.selected, np.where(mask), fig.selection_mode)
    fig.on_selection(lasso)