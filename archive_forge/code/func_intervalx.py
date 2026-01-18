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
@BboxBase.intervalx.setter
def intervalx(self, interval):
    self._points[:, 0] = interval
    self.invalidate()