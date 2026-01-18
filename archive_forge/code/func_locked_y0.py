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
@locked_y0.setter
def locked_y0(self, y0):
    self._locked_points.mask[0, 1] = y0 is None
    self._locked_points.data[0, 1] = y0
    self.invalidate()