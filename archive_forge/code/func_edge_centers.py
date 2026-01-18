from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
@property
def edge_centers(self):
    """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
    x0, y0, width, height = self._rect_bbox
    w = width / 2.0
    h = height / 2.0
    xe = (x0, x0 + w, x0 + width, x0 + w)
    ye = (y0 + h, y0, y0 + h, y0 + height)
    transform = self._get_rotation_transform()
    coords = transform.transform(np.array([xe, ye]).T).T
    return (coords[0], coords[1])