import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def juggle_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that 2D *xs*, *ys* can be plotted in the plane
    orthogonal to *zdir*. *zdir* is normally 'x', 'y' or 'z'. However, if
    *zdir* starts with a '-' it is interpreted as a compensation for
    `rotate_axes`.
    """
    if zdir == 'x':
        return (zs, xs, ys)
    elif zdir == 'y':
        return (xs, zs, ys)
    elif zdir[0] == '-':
        return rotate_axes(xs, ys, zs, zdir)
    else:
        return (xs, ys, zs)