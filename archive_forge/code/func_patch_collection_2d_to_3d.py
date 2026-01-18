import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def patch_collection_2d_to_3d(col, zs=0, zdir='z', depthshade=True):
    """
    Convert a `.PatchCollection` into a `.Patch3DCollection` object
    (or a `.PathCollection` into a `.Path3DCollection` object).

    Parameters
    ----------
    zs : float or array of floats
        The location or locations to place the patches in the collection along
        the *zdir* axis. Default: 0.
    zdir : {'x', 'y', 'z'}
        The axis in which to place the patches. Default: "z".
        See `.get_dir_vector` for a description of the values.
    depthshade
        Whether to shade the patches to give a sense of depth. Default: *True*.

    """
    if isinstance(col, PathCollection):
        col.__class__ = Path3DCollection
        col._offset_zordered = None
    elif isinstance(col, PatchCollection):
        col.__class__ = Patch3DCollection
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir)