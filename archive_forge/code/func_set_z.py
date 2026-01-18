import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def set_z(self, z):
    """
        Set the *z* position of the text.

        Parameters
        ----------
        z : float
        """
    self._z = z
    self.stale = True