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
@property
def minpos(self):
    """
        The minimum positive value in both directions within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum extent instead of *p0*.
        """
    return self._minpos