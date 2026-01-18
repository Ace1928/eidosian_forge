import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def set_data_3d(self, *args):
    """
        Set the x, y and z data

        Parameters
        ----------
        x : array-like
            The x-data to be plotted.
        y : array-like
            The y-data to be plotted.
        z : array-like
            The z-data to be plotted.

        Notes
        -----
        Accepts x, y, z arguments or a single array-like (x, y, z)
        """
    if len(args) == 1:
        args = args[0]
    for name, xyz in zip('xyz', args):
        if not np.iterable(xyz):
            raise RuntimeError(f'{name} must be a sequence')
    self._verts3d = args
    self.stale = True