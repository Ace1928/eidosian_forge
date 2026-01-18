import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def mutatedy(self):
    """Return whether the y-limits have changed since init."""
    return self._points[0, 1] != self._points_orig[0, 1] or self._points[1, 1] != self._points_orig[1, 1]