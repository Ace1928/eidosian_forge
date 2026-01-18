import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def set_rotate_label(self, val):
    """
        Whether to rotate the axis label: True, False or None.
        If set to None the label will be rotated if longer than 4 chars.
        """
    self._rotate_label = val
    self.stale = True