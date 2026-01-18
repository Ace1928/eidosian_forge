import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def line_collection_2d_to_3d(col, zs=0, zdir='z'):
    """Convert a `.LineCollection` to a `.Line3DCollection` object."""
    segments3d = _paths_to_3d_segments(col.get_paths(), zs, zdir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)